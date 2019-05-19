# Drowsy - a Bitsy game generator
<p align="center">
<img src="https://raw.githubusercontent.com/Pyrofoux/Drowsy/master/final/titleImage.png" width="40%" />
</p>

A Bitsy game generator, featuring  : 
 - a GAN to generate Avatars
 - a whole new Discriminator Network / Genetic Generator coupling to generate Rooms
 
 The AI generated game is available to [play here](https://pyrofoux.github.io/Drowsy/).
 
 
Here's the full [project report](https://github.com/Pyrofoux/Drowsy/raw/master/final/rapport.pdf) (French).
Below is an english summary.

*Made during 2nd year of engineering school, at the École Nationale Supérieure de Cognitique (Bordeaux, 2019)*


# Avatar generation 

## Dataset


<p align="center"><img src="https://raw.githubusercontent.com/Pyrofoux/Drowsy/master/final/eg-Avatar.png" /></p>

Ignoring color palette and animations, a Bitsy avatar is a 8x8 image with black and white pixels. The dataset is composed of 420 avatars, extracted from this compilation [tweet](https://twitter.com/ragzouken/status/939818949876830209).

A few examples from the dataset : 

<p align="center">
<img src="https://raw.githubusercontent.com/Pyrofoux/Drowsy/master/final/avatarDataset.png" />
</p>

## Generative Adversarial Network (GAN)

Here's the simplified structure of a GAN :

<p align="center">
<img src="https://raw.githubusercontent.com/Pyrofoux/Drowsy/master/final/gan.png" width="70%" />
</p>

The Antagonist is composed of the Discriminator network and a Generator network layered together.

<p align="center">
<img src="https://raw.githubusercontent.com/Pyrofoux/Drowsy/master/final/adversarialNetwork.png" width="40%" />
</p>

Since we process images, the chosen networks are CNNs.
You can find more information about GANs [here](https://skymind.ai/wiki/generative-adversarial-network-gan).

## Results

<p align="center">
<img src="https://raw.githubusercontent.com/Pyrofoux/Drowsy/master/final/avatarResults.png" width="40%" />
</p>

- **Data shape**

  The Generator cannot produce binary data, only reals in the range \[0.0 1.0].
  Produced avatars need a post-processing, to convert non-binary gray pixels to a black and white palette.
- **Aesthetic**

  The shapes do look like Bitsy avatars. They are evocative, allowing the player to give them a meaning. 
 
- **Diversity**

  Slight Mode Collapse around 2 or 3 classes at each training. Retraining just a few times switch the classes.
 - **Performance**
 
   These performances where obtained after 2 hours of training on a PC, with a GTX 1050 and 2 Go of RAM.  


# Rooms Generation

## Dataset

<p align="center"><img src="https://raw.githubusercontent.com/Pyrofoux/Drowsy/master/final/eg-Room.png" width="50%"  /></p>

Ignoring color palette, a Bitsy room is a 128x128 image with black and white pixels.
The dataset is composed of 591 rooms, extracted with a custom scrapper used on this [list](https://raw.githubusercontent.com/Pyrofoux/Drowsy/master/final/listOfGames.txt) of 21 Bitsy games.

## Preliminary Results

The initial approach was to use the same GAN structure than the Avatar generation, only scaled to process 128x128 images.

<p align="center">
<img src="https://raw.githubusercontent.com/Pyrofoux/Drowsy/master/final/mapResults1.png"  />
</p>

- **Data shape**

  Similarly, produced images need to be post-processed to only keep binary pixels. 
 
- **Diversity**

  Huge Mode Collapse.
 - **Performance**
 
   These performances where obtained after 4 hours of training on a Google Colaboratory dedicated machine, with GPU acceleration.  

- **Aesthetic**

  The shapes are structureless gradients, where Bitsy maps show these characteristics : 
  - local symmetry
  - pattern repetition
  - complex structures made of unitary pieces
 
## Setting up a new architecture : GADN

The unsatisfactory previous results led us to rethink the architecture behind Room generation.
Let's take another angle. We're looking for these characteristics : 
  - local symmetry
  - pattern repetition
  - complex structures made of unitary pieces
  
and they are commonly found in the realms of [Cellular Automata](https://en.wikipedia.org/wiki/Cellular_automaton).

<p align="center"><img src="https://raw.githubusercontent.com/Pyrofoux/Drowsy/master/final/caveGen.gif"  /></p>

*Example of cave-like level generation using cellular automata* ( [Source](https://codiecollinge.wordpress.com/2012/08/24/simple_2d_cave-like_generation) )


Instead of directly handling all the pixels to create a room, we could manipulate the rules of a Cellular Automaton that generates room-like images. The main issue is that the algorithm running a CA is exact, well known and simple whereas our current architecture is based on neural networks running complex and evolving algorithms, relevant for unpredictable cases.

There's a huge mismatch between the features we expect of the Antagonist, and it's nature. We need a structure both able to : 
- generate images following a specific algorithm (Cellular Automata)
- iteratively approach a criteria (fooling the Discriminator)

Our approach is to change the Antagonist from being a neural network to a [genetic algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm). The requirements to use such a structure are :
- being able to represent data as a set of **genes**
- having a way to evaluate the quality with a **fitness function**

In our case, the **genes** are the rules used to generate a room. They consist of the rules of a CA and the rules to generate the initial population of the grid (eg : the two parameters of a normal distribution). The key point is using the **evaluation** function of the Discriminator as the **fitness** function of our new Antagonist.

<p align="center">
<img src="https://raw.githubusercontent.com/Pyrofoux/Drowsy/master/final/coupling.png" width="70%" /></p>


## Final Results


