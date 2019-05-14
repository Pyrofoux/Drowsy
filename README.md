# Drowsy - Bitsy game generator
<p align="center">
<img src="https://raw.githubusercontent.com/Pyrofoux/Drowsy/master/final/titleImage.png" width="256px" height="256px" />
</p>

A Bitsy game generator, featuring  : 
 - a GAN to generate avatars
 - a state-of-the-art Discriminator Network / Genetic Generator coupling to generate rooms
 
 The AI generated game is available to [play here](https://pyrofoux.github.io/Drowsy/).
 
 
Here's the [project report](https://github.com/Pyrofoux/Drowsy/raw/master/final/rapport.pdf) (French).

*Made during 2nd year of engineering school, at the École Nationale Supérieure de Cognitique (Bordeaux, 2019)*


# Avatar generation 

## Dataset


<img src="https://raw.githubusercontent.com/Pyrofoux/Drowsy/master/final/eg-Avatar.png"  />

Ignoring color palette and animations, a Bitsy avatar is a 8x8 image with black and white pixels.
The dataset is composed of 420 avatars, extracted from this compilation [tweet](https://twitter.com/ragzouken/status/939818949876830209).

A few examples from the dataset : 

<img src="https://raw.githubusercontent.com/Pyrofoux/Drowsy/master/final/avatarDataset.png" />

## Generative Adversarial Network (GAN)


<img src="https://raw.githubusercontent.com/Pyrofoux/Drowsy/master/final/gan.png" width="70%" />



<img src="https://raw.githubusercontent.com/Pyrofoux/Drowsy/master/final/coupling.png" width="70%" />







