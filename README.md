# Drowsy - Bitsy game generator
<p align="center">
<img src="https://raw.githubusercontent.com/Pyrofoux/Drowsy/master/final/titleImage.png" width="256px" height="256px" />
</p>

A Bitsy game generator, featuring  : 
 - a GAN to generate avatars
 - a state-of-the-art Discriminator Network / Genetic Generator coupling to generate rooms
 
 The AI generated game is available to [play here](https://pyrofoux.github.io/Drowsy/).
 
 
Here's the full [project report](https://github.com/Pyrofoux/Drowsy/raw/master/final/rapport.pdf) (French).
Below is an english summary.

*Made during 2nd year of engineering school, at the École Nationale Supérieure de Cognitique (Bordeaux, 2019)*


# Avatar generation 

## Dataset


<img src="https://raw.githubusercontent.com/Pyrofoux/Drowsy/master/final/eg-Avatar.png"  />

Ignoring color palette and animations, a Bitsy avatar is a 8x8 image with black and white pixels.
The dataset is composed of 420 avatars, extracted from this compilation [tweet](https://twitter.com/ragzouken/status/939818949876830209).

A few examples from the dataset : 

<img src="https://raw.githubusercontent.com/Pyrofoux/Drowsy/master/final/avatarDataset.png" />

## Generative Adversarial Network (GAN)

Here's the simplified structure of a GAN :

<p align="center">
<img src="https://raw.githubusercontent.com/Pyrofoux/Drowsy/master/final/gan.png" width="70%" />
</p>

The Antagonist is composed of the Discriminator network and a Generator network layered together.

<p align="center">
<img src="https://raw.githubusercontent.com/Pyrofoux/Drowsy/master/final/adversarialNetwork.png" width="40%" />
</p>

You can find more information about GANs [here](https://skymind.ai/wiki/generative-adversarial-network-gan).

## Results

<img src="https://raw.githubusercontent.com/Pyrofoux/Drowsy/master/final/avatarResults.png" width="40%" />

- **Data shape**

  The Generator cannot produce binary data, only reals.
  Produced avatars need a post-processing, to convert gray pixels to a black and white palette.
- **Aesthetic**

  The shapes are evocative, allowing the player to imagine their meaning. 
 
- **Diversity*

 Slight Mode Collapse around 2 or 3 classes. Retraining a few times switch the classes.
 
 - **Performance**
 These performances where obtained after 2 hours of training, with a GTX 1050 and 2 Go of RAM.  



