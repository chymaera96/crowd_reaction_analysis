# Model recipes

The three primary architectures are BEATs, wav2vec2, and CNN–TCN. Each processes
16 kHz mono audio in 20-second bags and produces temporal crowd-event scores.
Joint models have independent event, approval, and disapproval heads.

## BEATs

A frozen BEATs iter3+ encoder, pretrained and fine-tuned on AudioSet-2M, encodes
one-second audio instances. Encoder tokens are averaged into 768-dimensional
features. Each task uses a 768→256→1 classifier with ReLU and dropout 0.1.
Output rate: 1 Hz.

## wav2vec2

A frozen `facebook/wav2vec2-base` encoder processes the continuous audio bag.
Waveforms are normalized by their mean and standard deviation. Transformer
layer 3 supplies 768-dimensional features, averaged into half-second bins.
Each task uses a 768→256→1 classifier with ReLU and dropout 0.1.
Output rate: 2 Hz.

## CNN–TCN

The frontend is a 64-bin log-mel spectrogram with a 25 ms window and 10 ms hop.
Spectrograms are normalized by their per-bag mean and standard deviation.

Three convolutional blocks use 16, 32, and 64 channels. Each block contains two
3×3 convolutions, GroupNorm, ReLU, pooling, and dropout 0.1. Frequency is reduced
in every block; total temporal downsampling is 4×. Four frequency bands are
retained and projected into a 64-dimensional temporal sequence.

Four residual temporal-convolution blocks use dilations 1, 2, 4, and 8, with two
convolutions per block. Each task has a 64→64→1 classifier. All parameters are
trained jointly. Output rate: 25 Hz, or one score every 40 ms.

The joint model has 200,467 trainable parameters. The event-only variant has
192,017 parameters and a single event head. It trains on the union of approval,
disapproval, and crowd-chorus labels; absence has probability `1 - event`.

## Training recipes

| Setting | BEATs | wav2vec2 | CNN–TCN |
|---|---|---|---|
| Encoder training | Frozen | Frozen | Trainable |
| Bag length | 20 s | 20 s | 20 s |
| Output bin | 1 s | 0.5 s | 0.04 s |
| Optimizer | AdamW | AdamW | AdamW |
| Learning rate | 1e-4 | 1e-4 | 1e-3 |
| Weight decay | 1e-4 | 1e-4 | 1e-4 |
| Batch size | 4 | 4 | 8 |
| Epochs | 25 | 25 | 40 |
| Seed | 0 | 0 | 0 |
| Weak loss | Max/mean/min | Max/mean/min | Linear-softmax BCE |
| Waveform augmentation | Disabled | Disabled | Lowpass, pink noise, clipping |
| Event threshold | 0.2 | 0.5 | 0.5 |
| Attribute threshold | 0.3 | 0.5 | 0.5 |
| Event onset collar | 1 s | 1 s | 1 s |
| Event offset ratio | 0.2 | 0.4 | 0.4 |

CNN–TCN augmentation applies lowpass filtering with probability 0.3 (3.5–7.5 kHz
cutoff), pink noise with probability 0.3 (10–25 dB SNR), and clipping with
probability 0.2. Validation audio is unaugmented.

## Weak supervision

Joint loss is `event loss + 0.5 × approval loss + 0.5 × disapproval loss`.
Attribute probabilities are conditioned on event probabilities. The event gate
is detached when computing attribute losses; the shared encoder receives the
combined task gradients.

Max/mean/min loss applies binary cross-entropy to the maximum, mean, and minimum
frame probabilities. Positive bags target maximum 1, mean 0.5, and minimum 0;
negative bags target zero for all three statistics. Linear-softmax pooling uses
`sum(p²) / sum(p)` followed by bag-level binary cross-entropy.

Clear polarity labels have weight 1 and unclear labels have weight 0.75.
Crowd chorus supervises the event head. No-crowd bags supervise event absence;
polarity losses are masked. Contradictory no-crowd and positive-event labels are
masked. Strong annotation files define the validation sources.

## Experimental notes

wav2vec2 experiments used intermediate transformer-layer outputs as frozen
features and also fused outputs from multiple layers. Fusion used a learned
scalar weight per layer, normalized with softmax, to form a weighted sum of
the temporally pooled layer representations before classification. Both
intermediate-layer selection and layer fusion gave suboptimal results in the
experiments conducted.

Both wav2vec2 and BEATs can learn speech presence or absence as a shortcut for
crowd activity. In particular, non-speech sections can be classified as crowd
reactions even when no reaction occurs. Inspect predictions across speech
pauses, silence, and background noise as well as actual applause, cheering, and
booing; speech gaps alone should not trigger an event.

The CNN–TCN transfers well to the archival-footage test recordings. Some
misclassifications are associated with changes in equalisation within the
recording, which alter the spectral balance of otherwise similar sounds.

## Future work

Explore encoders pretrained for paralinguistic information rather than speech
recognition, then fine-tune them with a similar weak-supervision recipe. Retain
the joint event and polarity objectives and compare their ability to distinguish
crowd reactions from speech pauses. Include archival recordings with varying
equalisation when assessing transfer, and investigate equalisation augmentation
to reduce sensitivity to recording conditions.
