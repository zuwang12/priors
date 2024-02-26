import os
import datetime
from concurrent import futures
import time
import ml_collections
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
import numpy as np
import ddpo_pytorch.prompts
import ddpo_pytorch.rewards
from ddpo_pytorch.stat_tracking import PerPromptStatTracker
from ddpo_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from ddpo_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
import torch
from torch.nn.utils import clip_grad_norm_
from functools import partial
import tqdm
import tempfile
from PIL import Image

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)
config = ml_collections.ConfigDict()

###### General ######
# run name for wandb logging and checkpoint saving -- if not provided, will be auto-generated based on the datetime.
config.run_name = ""
# random seed for reproducibility.
config.seed = 42
# top-level logging directory for checkpoint saving.
config.logdir = "logs"
# number of epochs to train for. each epoch is one round of sampling from the model followed by training on those
# samples.
config.num_epochs = 1
# number of epochs between saving model checkpoints.
config.save_freq = 20
# number of checkpoints to keep before overwriting old ones.
config.num_checkpoint_limit = 5
# mixed precision training. options are "fp16", "bf16", and "no". half-precision speeds up training significantly.
config.mixed_precision = "fp16"
# allow tf32 on Ampere GPUs, which can speed up training.
config.allow_tf32 = True
# resume training from a checkpoint. either an exact checkpoint directory (e.g. checkpoint_50), or a directory
# containing checkpoints, in which case the latest one will be used. `config.use_lora` must be set to the same value
# as the run that generated the saved checkpoint.
config.resume_from = ""
# whether or not to use LoRA. LoRA reduces memory usage significantly by injecting small weight matrices into the
# attention layers of the UNet. with LoRA, fp16, and a batch size of 1, finetuning Stable Diffusion should take
# about 10GB of GPU memory. beware that if LoRA is disabled, training will take a lot of memory and saved checkpoint
# files will also be large.
config.use_lora = False

config.is_local_main_process = True
config.num_processes = 1
config.process_index = 0
config.sync_gradients = True
config.is_main_process = True

###### Pretrained Model ######
config.pretrained = pretrained = ml_collections.ConfigDict()
# base model to load. either a path to a local directory, or a model name from the HuggingFace model hub.
pretrained.model = "runwayml/stable-diffusion-v1-5"
# revision of the model to load.
pretrained.revision = "main"

###### Sampling ######
config.sample = sample = ml_collections.ConfigDict()
# number of sampler inference steps.
sample.num_steps = 50
# eta parameter for the DDIM sampler. this controls the amount of noise injected into the sampling process, with 0.0
# being fully deterministic and 1.0 being equivalent to the DDPM sampler.
sample.eta = 1.0
# classifier-free guidance weight. 1.0 is no guidance.
sample.guidance_scale = 5.0
# batch size (per GPU!) to use for sampling.
sample.batch_size = 1
# number of batches to sample per epoch. the total number of samples per epoch is `num_batches_per_epoch *
# batch_size * num_gpus`.
sample.num_batches_per_epoch = 2

###### Training ######
config.train = train = ml_collections.ConfigDict()
# batch size (per GPU!) to use for training.
train.batch_size = 1
# whether to use the 8bit Adam optimizer from bitsandbytes.
train.use_8bit_adam = False
# learning rate.
train.learning_rate = 3e-4
# Adam beta1.
train.adam_beta1 = 0.9
# Adam beta2.
train.adam_beta2 = 0.999
# Adam weight decay.
train.adam_weight_decay = 1e-4
# Adam epsilon.
train.adam_epsilon = 1e-8
# number of gradient accumulation steps. the effective batch size is `batch_size * num_gpus *
# gradient_accumulation_steps`.
train.gradient_accumulation_steps = 1
# maximum gradient norm for gradient clipping.
train.max_grad_norm = 1.0
# number of inner epochs per outer epoch. each inner epoch is one iteration through the data collected during one
# outer epoch's round of sampling.
train.num_inner_epochs = 1
# whether or not to use classifier-free guidance during training. if enabled, the same guidance scale used during
# sampling will be used during training.
train.cfg = True
# clip advantages to the range [-adv_clip_max, adv_clip_max].
train.adv_clip_max = 5
# the PPO clip range.
train.clip_range = 1e-4
# the fraction of timesteps to train on. if set to less than 1.0, the model will be trained on a subset of the
# timesteps for each sample. this will speed up training but reduce the accuracy of policy gradient estimates.
train.timestep_fraction = 1.0

###### Prompt Function ######
# prompt function to use. see `prompts.py` for available prompt functions.
config.prompt_fn = "imagenet_animals"
# kwargs to pass to the prompt function.
config.prompt_fn_kwargs = {}

###### Reward Function ######
# reward function to use. see `rewards.py` for available reward functions.
config.reward_fn = "tsp"
# config.reward_fn = "jpeg_compressibility"

###### Per-Prompt Stat Tracking ######
# when enabled, the model will track the mean and std of reward on a per-prompt basis and use that to compute
# advantages. set `config.per_prompt_stat_tracking` to None to disable per-prompt stat tracking, in which case
# advantages will be calculated using the mean and std of the entire batch.
config.per_prompt_stat_tracking = ml_collections.ConfigDict()
# number of reward values to store in the buffer for each prompt. the buffer persists across epochs.
config.per_prompt_stat_tracking.buffer_size = 16
# the minimum number of reward values to store in the buffer before using the per-prompt mean and std. if the buffer
# contains fewer than `min_count` values, the mean and std of the entire batch will be used instead.
config.per_prompt_stat_tracking.min_count = 16

torch.cuda.set_device(2)

unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
if not config.run_name:
    config.run_name = unique_id
else:
    config.run_name += "_" + unique_id

if config.resume_from:
    config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
    if "checkpoint_" not in os.path.basename(config.resume_from):
        # get the most recent checkpoint in this directory
        checkpoints = list(
            filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from))
        )
        if len(checkpoints) == 0:
            raise ValueError(f"No checkpoints found in {config.resume_from}")
        config.resume_from = os.path.join(
            config.resume_from,
            sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
        )

# number of timesteps within each trajectory to train on
num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

# accelerator_config = ProjectConfiguration(
#     project_dir=os.path.join(config.logdir, config.run_name),
#     automatic_checkpoint_naming=True,
#     total_limit=config.num_checkpoint_limit,
# )

# accelerator = Accelerator(
#     log_with="wandb",
#     mixed_precision=config.mixed_precision,
#     project_config=accelerator_config,
#     # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
#     # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
#     # the total number of optimizer steps to accumulate across.
#     gradient_accumulation_steps=config.train.gradient_accumulation_steps
#     * num_train_timesteps,
# )

# set seed (device_specific is very important to get different prompts on different devices)
# set_seed(config.seed, device_specific=True)

# load scheduler, tokenizer and models.
pipeline = StableDiffusionPipeline.from_pretrained(
    config.pretrained.model, revision=config.pretrained.revision # main
)
# freeze parameters of models to save more memory
pipeline.vae.requires_grad_(False)
pipeline.text_encoder.requires_grad_(False)
pipeline.unet.requires_grad_(not config.use_lora) # True
# disable safety checker
pipeline.safety_checker = None
# make the progress bar nicer
pipeline.set_progress_bar_config(
    position=1,
    disable=not config.is_local_main_process,
    leave=False,
    desc="Timestep",
    dynamic_ncols=True,
)
# switch to DDIM scheduler
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

# For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
# as these weights are only used for inference, keeping weights in full precision is not required.
inference_dtype = torch.float16

# Move unet, vae and text_encoder to device and cast to inference_dtype
pipeline.vae.to(torch.device("cuda"), dtype=inference_dtype)
pipeline.text_encoder.to(torch.device("cuda"), dtype=inference_dtype)
unet = pipeline.unet.to(torch.device("cuda"), dtype=inference_dtype)
    
# set up diffusers-friendly checkpoint saving with Accelerate

# Enable TF32 for faster training on Ampere GPUs,
# cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
if config.allow_tf32:
    torch.backends.cuda.matmul.allow_tf32 = True

# Initialize the optimizer
if config.train.use_8bit_adam:
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError(
            "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
        )

    optimizer_cls = bnb.optim.AdamW8bit
else:
    optimizer_cls = torch.optim.AdamW

optimizer = optimizer_cls(
    unet.parameters(),
    lr=config.train.learning_rate,
    betas=(config.train.adam_beta1, config.train.adam_beta2),
    weight_decay=config.train.adam_weight_decay,
    eps=config.train.adam_epsilon,
)

# prepare prompt and reward fn
prompt_fn = getattr(ddpo_pytorch.prompts, config.prompt_fn)
reward_fn = getattr(ddpo_pytorch.rewards, config.reward_fn)()

# generate negative prompt embeddings
neg_prompt_embed = pipeline.text_encoder(
    pipeline.tokenizer(
        [""],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=pipeline.tokenizer.model_max_length,
    ).input_ids.to(torch.device("cuda"))
)[0]
sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.batch_size, 1, 1)
train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)

# initialize stat tracker
if config.per_prompt_stat_tracking:
    stat_tracker = PerPromptStatTracker(
        config.per_prompt_stat_tracking.buffer_size,
        config.per_prompt_stat_tracking.min_count,
    )

# executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
# remote server running llava inference.
executor = futures.ThreadPoolExecutor(max_workers=2)

# Train!



samples_per_epoch = (
    config.sample.batch_size
    * config.num_processes
    * config.sample.num_batches_per_epoch
)
total_train_batch_size = (
    config.train.batch_size
    * config.num_processes
    * config.train.gradient_accumulation_steps
)

assert config.sample.batch_size >= config.train.batch_size
assert config.sample.batch_size % config.train.batch_size == 0
assert samples_per_epoch % total_train_batch_size == 0

if config.resume_from:
    # accelerator.load_state(config.resume_from)
    first_epoch = int(config.resume_from.split("_")[-1]) + 1
else:
    first_epoch = 0

global_step = 0
for epoch in range(first_epoch, config.num_epochs):
    #################### SAMPLING ####################
    pipeline.unet.eval()
    samples = []
    prompts = []
    for i in tqdm(
        range(config.sample.num_batches_per_epoch), # range(2)
        desc=f"Epoch {epoch}: sampling",
        disable=not config.is_local_main_process,
        position=0,
    ):
        # generate prompts
        prompts, prompt_metadata = zip(
            *[
                prompt_fn(**config.prompt_fn_kwargs)
                for _ in range(config.sample.batch_size)
            ]
        )

        # encode prompts
        prompt_ids = pipeline.tokenizer(
            prompts, # ('starfish, sea star',)
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(torch.device("cuda")) # -> [1, 77]
        prompt_embeds = pipeline.text_encoder(prompt_ids)[0] # [1, 77, 768]

        # sample
        images, _, latents, log_probs = pipeline_with_logprob(
            pipeline, # StableDiffusionPipeline
            prompt_embeds=prompt_embeds, # [1, 77, 768]
            negative_prompt_embeds=sample_neg_prompt_embeds, # [1, 77, 768] TODO: 이거가 uncond_embedding인가?
            num_inference_steps=config.sample.num_steps, # 50 | TODO: diffusion step 말하는 건가?
            guidance_scale=config.sample.guidance_scale, # 5.0
            eta=config.sample.eta, # 1.0
            output_type="pt",
        ) #[1, 3, 512, 512], 51 x [1, 4, 64, 64], 50 TODO: 이거 Output인가?

        latents = torch.stack(
            latents, dim=1
        )  # (batch_size, num_steps + 1, 4, 64, 64) ~ ([1, 51, 4, 64, 64])
        log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1) ~ ([1, 50])
        timesteps = pipeline.scheduler.timesteps.repeat(
            config.sample.batch_size, 1
        )  # (batch_size, num_steps) ~ ([1, 50])

        # compute rewards asynchronously
        points = np.load('./points.npy') #TODO: need to modify
        rewards = executor.submit(reward_fn, points) # , , ('starfish, sea star',), ({}, )
        # rewards = executor.submit(reward_fn, images, prompts, prompt_metadata) # , , ('starfish, sea star',), ({}, )
        # yield to to make sure reward computation starts
        time.sleep(0)

        samples.append(
            {
                "prompt_ids": prompt_ids, # [1, 77]
                "prompt_embeds": prompt_embeds, # [1, 77, 768]
                "timesteps": timesteps, # [1, 50]
                "latents": latents[
                    :, :-1
                ],  # each entry is the latent before timestep t -> ([1, 50, 4, 64, 64])
                "next_latents": latents[
                    :, 1:
                ],  # each entry is the latent after timestep t -> ([1, 50, 4, 64, 64])
                "log_probs": log_probs, # [1, 50]
                "rewards": rewards,
            }
        )

    # wait for all rewards to be computed
    for sample in tqdm(
        samples,
        desc="Waiting for rewards",
        disable=not config.is_local_main_process,
        position=0,
    ):
        rewards, reward_metadata = sample["rewards"].result() # -137.345, {} | TODO: 뭐하는 거..?
        sample["rewards"] = torch.as_tensor(rewards, device=torch.device("cuda"))

    # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
    samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}

    # this is a hack to force wandb to log the images as JPEGs instead of PNGs
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, image in enumerate(images):
            pil = Image.fromarray(
                (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            )
            pil = pil.resize((256, 256))
            pil.save(os.path.join(tmpdir, f"{i}.jpg"))


    # gather rewards across processes
    rewards = samples["rewards"].cpu().numpy()
    
    # per-prompt mean/std tracking
    if config.per_prompt_stat_tracking:
        # gather the prompts across processes
        prompt_ids = samples["prompt_ids"].cpu().numpy() # (2, 77)
        prompts = pipeline.tokenizer.batch_decode(
            prompt_ids, skip_special_tokens=True
        ) # ['starfish, sea star', 'garter snake, grass snake']
        advantages = stat_tracker.update(prompts, rewards) # array([ 0.9999998, -0.9999998])
    else:
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # ungather advantages; we only need to keep the entries corresponding to the samples on this process
    samples["advantages"] = (
        torch.as_tensor(advantages) # ([2])
        .reshape(config.num_processes, -1)[config.process_index]
        .to(torch.device("cuda"))
    ) # tensor([ 1.0000, -1.0000])

    del samples["rewards"]
    del samples["prompt_ids"]

    total_batch_size, num_timesteps = samples["timesteps"].shape # (2, 50)
    assert (
        total_batch_size # 2
        == config.sample.batch_size * config.sample.num_batches_per_epoch # 1 * 2
    )
    assert num_timesteps == config.sample.num_steps # 50

    #################### TRAINING ####################
    for inner_epoch in range(config.train.num_inner_epochs): # in range(1)
        # shuffle samples along batch dimension
        perm = torch.randperm(total_batch_size, device=torch.device("cuda")) # torch.randperm(2)
        samples = {k: v[perm] for k, v in samples.items()}

        # shuffle along time dimension independently for each sample
        perms = torch.stack(
            [
                torch.randperm(num_timesteps, device=torch.device("cuda"))
                for _ in range(total_batch_size)
            ]
        ) # [2, 50]
        for key in ["timesteps", "latents", "next_latents", "log_probs"]:
            samples[key] = samples[key][
                torch.arange(total_batch_size, device=torch.device("cuda"))[:, None],
                perms,
            ]

        # rebatch for training : [2, 50, 4, 64, 64] -> [2, 1, 50, 4, 64, 64]
        samples_batched = {
            k: v.reshape(-1, config.train.batch_size, *v.shape[1:])
            for k, v in samples.items()
        }

        # dict of lists -> list of dicts for easier iteration
        samples_batched = [
            dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
        ]

        # train
        pipeline.unet.train()
        # info = defaultdict(list)
        for i, sample in tqdm(
            list(enumerate(samples_batched)),
            desc=f"Epoch {epoch}.{inner_epoch}: training",
            position=0,
            disable=not config.is_local_main_process,
        ):
            if config.train.cfg:
                # concat negative prompts to sample prompts to avoid two forward passes
                embeds = torch.cat(
                    [train_neg_prompt_embeds, sample["prompt_embeds"]] # [1, 77, 768], [1, 77, 768] -> [2, 77, 768]
                )
            else:
                embeds = sample["prompt_embeds"]

            for j in tqdm(
                range(num_train_timesteps), # range(50)
                desc="Timestep",
                position=1,
                leave=False,
                disable=not config.is_local_main_process,
            ):
                if config.train.cfg:
                    noise_pred = unet(
                        torch.cat([sample["latents"][:, j]] * 2), # [1, 50, 4, 64, 64] -> [2, 4, 64, 64]
                        torch.cat([sample["timesteps"][:, j]] * 2), # [1, 50] -> [2]
                        embeds, # [2, 77, 768]
                    ).sample
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2) # [2, 4, 64, 64] -> [1, 4, 64, 64], [1, 4, 64, 64]
                    noise_pred = ( # 이 부분 다시 check
                        noise_pred_uncond
                        + config.sample.guidance_scale
                        * (noise_pred_text - noise_pred_uncond)
                    )
                else:
                    noise_pred = unet(
                        sample["latents"][:, j],
                        sample["timesteps"][:, j],
                        embeds,
                    ).sample
                # compute the log prob of next_latents given latents under the current model
                _, log_prob = ddim_step_with_logprob(
                    pipeline.scheduler,
                    noise_pred, # model output
                    sample["timesteps"][:, j],
                    sample["latents"][:, j],
                    eta=config.sample.eta,
                    prev_sample=sample["next_latents"][:, j],
                )

                # ppo logic
                advantages = torch.clamp(
                    sample["advantages"],
                    -config.train.adv_clip_max,
                    config.train.adv_clip_max,
                )
                ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                unclipped_loss = -advantages * ratio
                clipped_loss = -advantages * torch.clamp(
                    ratio,
                    1.0 - config.train.clip_range,
                    1.0 + config.train.clip_range,
                )
                loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                # debugging values
                # John Schulman says that (ratio - 1) - log(ratio) is a better
                # estimator, but most existing code uses this so...
                # http://joschu.net/blog/kl-approx.html

                loss.backward()
                if config.sync_gradients:
                    clip_grad_norm_(
                        unet.parameters(), config.train.max_grad_norm
                    )
                optimizer.step()
                optimizer.zero_grad()

        # make sure we did an optimization step at the end of the inner epoch
        assert config.sync_gradients

    # if epoch != 0 and epoch % config.save_freq == 0 and config.is_main_process:
    #     accelerator.save_state()