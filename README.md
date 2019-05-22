# Drowsy - Making an AI making tiny Bitsy video games
<p align="center">
<img src="https://raw.githubusercontent.com/Pyrofoux/Drowsy/master/final/titleImage.png" width="40%" />
</p>

**Drowsy** is a school project aiming to build an AI architecture able to autonomously create a video game compatible with the [Bitsy](https://github.com/le-doux/bitsy) engine.

It features  : 
 - a GAN to generate Avatars
 - a whole new Discriminator Network / Adversarial Genetic Algorithm coupling to generate Rooms ([learn more](#building-a-new-architecture--dnaga-coupling))
 
 The AI generated game is available to [play here](https://pyrofoux.github.io/Drowsy/).
 
 
Here's the full [project report](https://github.com/Pyrofoux/Drowsy/raw/master/final/rapport.pdf) (French).
Below is an english summary.


For any questions, contact me at <yrabii@ensc.fr>, or on [Twitter](https://twitter.com/Pyrofoux). I'm currently looking for opportunities to study AI and Computational Creativity, especially a PhD. Please get in touch !

*Made in  3 months, during 2nd year of engineering school, at the École Nationale Supérieure de Cognitique (Bordeaux, 2019)*


Table of Contents
=================

   * [Avatar generation](#avatar-generation)
      * [Dataset](#dataset)
      * [Generative Adversarial Network (GAN)](#generative-adversarial-network-gan)
      * [Results](#results)
   * [Rooms Generation](#rooms-generation)
      * [Dataset](#dataset-1)
      * [Preliminary Results](#preliminary-results)
      * [Building a new architecture : DN/AGA coupling](#building-a-new-architecture--dnaga-coupling)
      * [Final Results](#final-results)
   * [Conclusion](#conclusion)
      * [Making the final game](#making-the-final-game)
      * [Studying the DN/AGA coupling](#studying-the-dnaga-coupling)

# Avatar generation 

## Dataset


<p align="center"><img src="https://raw.githubusercontent.com/Pyrofoux/Drowsy/master/final/eg-Avatar.png" />
<br/><i>Bitsy avatar, representing a human shape</i></p>

Ignoring color palette and animations, a Bitsy avatar is a 8x8 image with black and white pixels. The dataset is composed of 420 avatars, extracted from this compilation [tweet](https://twitter.com/ragzouken/status/939818949876830209).


<p align="center">
<img src="https://raw.githubusercontent.com/Pyrofoux/Drowsy/master/final/avatarDataset.png" /><br/><i>Examples from the avatars dataset</i></p>

## Generative Adversarial Network (GAN)

<p align="center">
<img src="https://raw.githubusercontent.com/Pyrofoux/Drowsy/master/final/gan.png" width="70%" /><br/><i>Simplified structure of a GAN</i><br/><i></i></p>

The Adversarial part is composed of the Discriminator network and a Generator network layered together.

<p align="center">
<img src="https://raw.githubusercontent.com/Pyrofoux/Drowsy/master/final/adversarialNetwork.png" width="40%" />
</p>

Since we process images, the chosen networks are CNNs.
You can find more information about GANs [here](https://skymind.ai/wiki/generative-adversarial-network-gan).

## Results

<p align="center">
<img src="https://raw.githubusercontent.com/Pyrofoux/Drowsy/master/final/avatarResults.png" width="40%" /><br/><i>Outputs of the avatar GAN after 2 hours of training</i></p>

- **Data shape**

  The Generator cannot produce binary data, only reals in the range \[0.0 - 1.0].
  Produced avatars need a post-processing, to convert non-binary gray pixels to a black and white palette.
- **Aesthetic**

  The shapes do look like Bitsy avatars. They are evocative, allowing the player to give them a meaning. 
 
- **Diversity**

  Slight Mode Collapse around 2 or 3 classes at each training. Retraining just a few times switch the classes.
 - **Performance**
 
   These performances where obtained after 2 hours of training on a laptop, with a GTX 1050 and 2 Go of RAM.  


# Rooms Generation

## Dataset

<p align="center"><img src="https://raw.githubusercontent.com/Pyrofoux/Drowsy/master/final/eg-Room.png" width="50%"  /><br/><i>typical Bitsy room, representing a forest (<a href="https://ledoux.itch.io/in-the-middle-of-the-night">Source</a>)</i></p>

Ignoring color palette, a Bitsy room is a 128x128 image with black and white pixels.
The dataset is composed of 591 rooms, extracted with a custom scrapper used on this [list](https://raw.githubusercontent.com/Pyrofoux/Drowsy/master/final/listOfGames.txt) of 21 Bitsy games.

## Preliminary Results

The initial approach was to use the same GAN structure than the Avatar generation, only scaled to process 128x128 images.

<p align="center">
<img src="https://raw.githubusercontent.com/Pyrofoux/Drowsy/master/final/mapResults1.png"  /><br/><i>Outputs of the Room GAN after 4 hours of training</i></p>

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
 
## Building a new architecture : DN/AGA coupling

The unsatisfactory previous results led us to rethink the architecture behind Room generation.
Let's take another angle. We're looking for these characteristics : 
  - local symmetry
  - pattern repetition
  - complex structures made of unitary pieces
  
and they are commonly found in the realms of [Cellular Automata](https://en.wikipedia.org/wiki/Cellular_automaton).

<div align="center"> <img src="https://raw.githubusercontent.com/Pyrofoux/Drowsy/master/final/caveGen.gif"  />

*Example of cave-like level generation using cellular automata* ([Source](https://codiecollinge.wordpress.com/2012/08/24/simple_2d_cave-like_generation))
</div>

Cellular Automata are frequently used to procedurally generate video game levels. They iteratively apply local transformation rules to an initial grid until it has specific characteristics. Instead of directly handling all the pixels to create a room, we could manipulate the rules of a Cellular Automaton that generates room-like images.

The main issue is that the algorithm running a CA is exact, well known and simple whereas our current architecture is based on neural networks running complex and evolving algorithms, relevant for unpredictable cases.

There's a huge mismatch between the features we expect of the Adversarial half, and it's nature. We need a structure both able to : 
- generate images following a specific algorithm (Cellular Automata)
- iteratively approach a criteria (fooling the Discriminator)

Our approach is to change the Adversarial half from being a neural network to a [genetic algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm). The requirements to use such a structure are :
- being able to represent data as a set of **genes**
- having a way to evaluate their quality with a **fitness function**

In our case, the **genes** are the rules used to generate a room. They consist of the rules of a CA and the rules to generate the initial population of the grid (eg : the two parameters of a normal distribution). The key point is using the **evaluation** function of the Discriminator as the **fitness** function of our new Adversarial component.

<p align="center">
<img src="https://raw.githubusercontent.com/Pyrofoux/Drowsy/master/final/coupling.png" width="70%" /><br/><i>Structure of the DN/AGA coupling</i></p>

If we see a GAN as a coupling between a Discriminator Network and an Antagonist Network (DN/AN), we can describe our new architecture as an asymmetric coupling between a Discriminator Network and an Adversarial Genetic Algorithm (DN/AGA).




## Final Results

<div align="center"><img src="https://raw.githubusercontent.com/Pyrofoux/Drowsy/master/final/roomResult1.png" width="45%" /> <img src="https://raw.githubusercontent.com/Pyrofoux/Drowsy/master/final/roomResult2.png" width="45%" /><img src="https://raw.githubusercontent.com/Pyrofoux/Drowsy/master/final/roomResult3.png" width="45%" /> <img src="https://raw.githubusercontent.com/Pyrofoux/Drowsy/master/final/roomResult4.png" width="45%" />

<i> Rooms generated by CA rules selected by the DN/AGA coupling<br/>
</i>
</div>


- **Data shape**

  The generated rooms can directly be used a Bitsy rooms, without post-processing.
 
- **Diversity**

  The selected room rules by the AGA are diverse, there's no observed phenomenon equivalent to Mode Collapse. One selected set of rules can be reused multiple times to produce rooms with similar features, while being a  different room. Simply changing the rules behind the generation of the CA initial population can heavily impact the style of the room . Eg: from Normal Distribution (*down right room*) to Uniform Distribution (*top left room*).
 - **Performance**
 
   These performances where obtained after 2 hours of training on a PC, with a GTX 1050 and 2 Go of RAM. (Same machine as in the Avatar generation)

- **Aesthetic**

  We observe the desired characteristics : 
  - local symmetry
  - pattern repetition
  - complex structures made of unitary pieces
  
  Furthermore, the generated rooms reproduce structures often found in Bitsy games. 
  - top right : maze-like room, made of several corridors
  - top left : museum-like room, with distinct items scattered in the space
  - bottom left : road leading to another room
  - bottom right : huge central structure
  
  More examples can be found in the [archives](https://github.com/Pyrofoux/Drowsy/tree/master/archives).


# Conclusion

## Making the final game

The [final game](https://pyrofoux.github.io/Drowsy/) was made by compiling together a selection of rooms generated by several cellular automata, whose rules were designed after 1 hour of training by the DN/AGA coupling. The player's avatar was generated by a GAN, after 2 hours of training. We then manually added doors to go from one room to another, as well as an introduction and conclusion text.


## Studying the DN/AGA coupling

The DN/AGA coupling was experimentally developed to go beyond the constraint of having a Generator in the form of a neural network. This structure generalizes the theory behind classical GANs and extends its scope of application to fields where the use of explicit algorithms is relevant. To our knowledge, this kind of coupling between a Discriminator Network and an Adversarial Genetic Algorithm is yet to be studied academically.

Here are some observations we made empirically : 
- **Costs**

	The DN/AGA structure appears to be way less costly in computation time than its GAN counterpart, and can be done with cheaper hardware
- **Training**
	
    Classical GAN training is made by alternating turns : training the DN on a batch, then the AN, and repeat. Better results were obtained for the DN/AGA by letting each part train enough times before handing over to the other.  Criteria for determining if more training was needed were : 
    - for the Discriminator Network : an accuracy below a specific threshold
    - for the Adversarial Genetic Algorithm : the median fitness of the population below a specific threshold 


For any questions, contact me at <yrabii@ensc.fr>, or on [Twitter](https://twitter.com/Pyrofoux). I'm currently looking for opportunities to study AI and Computational Creativity, especially a PhD. Please get in touch !
