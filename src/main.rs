use anyhow::{Error as E, Result};
use clap::Parser;

use candle_transformers::models::mixformer::{Config, MixFormerSequentialForCausalLM as MixFormer};
use candle_transformers::models::phi::Model as Phi;
use candle_transformers::models::quantized_mixformer::MixFormerSequentialForCausalLM as QMixFormer;

use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

enum Model {
    MixFormer(MixFormer),
    Phi(Phi),
    Quantized(QMixFormer),
}

struct TextGeneration {
    model: Model,
    device: Device,
    tokenizer: Tokenizer,
    eos_token: u32,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

struct TextGenerationIterator {
    tokens: Vec<u32>,
    generation: TextGeneration,
    max_len: usize,
    index: usize,
    generated_tokens: usize,
    start_gen: std::time::Instant,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        temp: f64,
        top_p: f64,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Result<Self> {
        let logits_processor = LogitsProcessor::new(seed, Some(temp), Some(top_p));

        let mut generated_tokens = 0usize;
        let eos_token = match tokenizer.get_vocab(true).get("<|endoftext|>") {
            Some(token) => *token,
            None => anyhow::bail!("cannot find the endoftext token"),
        };

        Ok(Self {
            model,
            tokenizer,
            eos_token,
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        })
    }
    fn run(self, prompt: &str, max_len: usize) -> Result<TextGenerationIterator> {
        let tokens = self.tokenizer.encode(prompt, true).map_err(E::msg)?;
        if tokens.is_empty() {
            anyhow::bail!("Empty prompts are not supported in the phi model.")
        }
        let tokens = tokens.get_ids().to_vec();

        return TextGenerationIterator::new(self, tokens, max_len);
    }
}

impl TextGenerationIterator {
    fn new(generation: TextGeneration, tokens: Vec<u32>, max_len: usize) -> Result<Self> {
        let start_gen = std::time::Instant::now();
        Ok(Self {
            tokens,
            generation,
            max_len,
            index: 0,
            generated_tokens: 0,
            start_gen,
        })
    }

    fn gen_token(&mut self) -> Result<u32> {
        let context_size = if self.index > 0 { 1 } else { self.tokens.len() };

        let ctxt = &self.tokens[self.tokens.len().saturating_sub(context_size)..];

        let input = Tensor::new(ctxt, &self.generation.device)?.unsqueeze(0)?;

        let logits = match &mut self.generation.model {
            Model::MixFormer(m) => m.forward(&input)?,
            Model::Phi(m) => m.forward(&input)?,
            Model::Quantized(m) => m.forward(&input)?,
        };
        let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;

        let logits = if self.generation.repeat_penalty == 1. {
            logits
        } else {
            let start_at = self
                .tokens
                .len()
                .saturating_sub(self.generation.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                self.generation.repeat_penalty,
                &self.tokens[start_at..],
            )?
        };

        let next_token = self.generation.logits_processor.sample(&logits)?;

        self.tokens.push(next_token);

        self.generated_tokens += 1;

        return Ok(next_token);
    }
}

impl Iterator for TextGenerationIterator {
    type Item = Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.generated_tokens >= self.max_len {
            let dt = self.start_gen.elapsed();
            let generated_tokens = self.generated_tokens;
            println!(
                "\n{generated_tokens} tokens generated ({:.2} token/s)",
                self.generated_tokens as f64 / dt.as_secs_f64(),
            );
            return None;
        }

        match self.gen_token() {
            Ok(next_token) => {
                if next_token == self.generation.eos_token {
                    return None;
                }
                println!("Got token: {next_token}");
                return Some(
                    self.generation
                        .tokenizer
                        .decode(&[next_token], true)
                        .map_err(E::msg),
                );
            }
            Err(e) => {
                return Some(Err(e));
            }
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// The temperature used to generate samples.
    #[arg(long, default_value_t = 0.3)]
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    #[arg(long, default_value_t = 0.95)]
    top_p: f64,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 5000)]
    max_len: usize,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
}

fn init_generator(args: Args) -> Result<TextGeneration> {
    let api = Api::new()?;

    let revision = "main".to_string();

    let model_id = "lmz/candle-quantized-phi".to_string();

    let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));

    let tokenizer_filename = repo.get("tokenizer.json")?;
    let filenames = vec![repo.get("model-v2-q4k.gguf")?];

    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let device = Device::Cpu;
    let config = Config::v2();

    println!("Model: {:?}", filenames[0]);

    let vb =
        candle_transformers::quantized_var_builder::VarBuilder::from_gguf(&filenames[0], &device)?;
    let model = Model::Quantized(QMixFormer::new_v2(&config, vb)?);

    return TextGeneration::new(
        model,
        tokenizer,
        args.seed,
        args.temperature,
        args.top_p,
        args.repeat_penalty,
        args.repeat_last_n,
        &device,
    );
}

fn summarize(text: String, pipeline: TextGeneration, max_len: usize) -> Result<Vec<String>> {
    use std::io::Write;
    let mut summaries = Vec::new();

    let mut prompt: String = r#"Instruct: ""#.to_owned();

    let end_prompt: &str = r#""

Summarize up to 10 key points from the group coversation above in a markdown list, then add a separate list for "action items" and "dates".
Output: "#;

    prompt.push_str(&text);
    prompt.push_str(end_prompt);

    for result in pipeline.run(&prompt, max_len)? {
        let token = result?;
        print!("{token}");
        std::io::stdout().flush()?;
    }

    return Ok(summaries);
}

fn main() -> Result<()> {
    let args = Args::parse();
    let max_len = args.max_len;

    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.temperature, args.repeat_penalty, args.repeat_last_n
    );

    let mut pipeline = init_generator(args)?;

    let inputt = r#"Okay, so it's not video.
It's just audio and then hopefully I can just rip the audio out of it.
My partners, Vex's folks are coming by for the week and so today we're in like a house cleaning rush to like make it presentable.
They're not staying over here the whole week, which is like, oh, this dress.
We're not like the most tidy people so I'm like, oh, you got all like clean stuff.
Yeah, but let me find.
Yeah, how are you doing? AccuLash? I'm doing good.
It's kind of hard to focus because it's wedding season.
Yeah, that's it.
The old man is quite excited.
I don't know if you can see.
Hey, he's on the phone.
Oh, are you on your phone? .
I'm in this small family adventure.
I wanted to let you know that I am repairing my computer.
So I'm using like an old D1 and it's not very fast.
So I've been trying to take notes in HackMD and it's just unresponsive.
So maybe I will be able to take notes if someone else could do it.
I can take notes.
We can also turn off our video to save.
We can also try turning off our video to save on computing resources.
Because that usually just really uses up a lot of system.
If it's really bad, let us know.
So begin with, I'll be okay with video because I like saying like everybody's expressions.
So I'll let you know if I need it or I've changed my own performance.
There's just a couple of things.
I'm not sure if I'm going to do it.
I'm going to do it.
I'm going to do it .
I can't do anything.
The social impact newsletter is going out again next week.
And so Bryn emailed about sending her updates to add to that by tomorrow.
I was just going to write a short thing about the social inbox release.
Since we hadn't sent that out officially on that channel.
Is there anything else that we want to add to that social impact newsletter update? Maybe a teaser that we're creating a reader for loading .
So I'll draft that today and I'll send that around in the chat for a plus one from folks.
And then let's see.
And then the social media post brainstorm.
So I wanted to just post more on our channels, which is pretty much the fediverse.
I'm personally kind of boycotting Twitter and X.
I don't know if we want to if we want to do that as distributed press.
We technically want to do that.
We want to do that.
We want to .
But I wanted to see if there was any other like, you know, little posts that we could do.
As I'm starting to do research about the FEDERverse threads integration, I was going to maybe post like, I mean, I think everybody in the FEDERverse is pretty like in the know about a lot of the stuff, but there was a meeting about, um, I think that's a really interesting thing about the FEDERverse.
I think that's a really interesting thing about the FEDERverse.
I think that's a really interesting thing .
I think it would be really cool if we had a primer that's for non-federal folks as well.
Like, why do we care about this outside of heck people? I feel like maybe something that compost readers in particular could understand would be real.
I think that's a really cool thing.
I think that we should think of what type of format, like amount of words or what type of piece we were.
We're not going to do that.
We're not going to .
And maybe we can have a teaser for that too and say this is coming, but then have this thread ready for when it's available.
I was thinking more to start thinking about it before the release so that we're not as rushed.
I think teasers are a great idea, maybe even for the newsletter."#.to_string();

    let input =
        "I got you a hamburger but I eated it. :( I will get you another one tonight. Sorry!"
            .to_string();

    let summaries = summarize(input, pipeline, max_len)?;

    println!("Summaries: {:?}", summaries);

    Ok(())
}
