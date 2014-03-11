Proto-Net
========
This project is a barebonese C# Neural Network designed to be an example of how to implement backpropagation.
My goal was to heavily comment the code and make it as logical as possible. There is a fair amount of recursion
and, although it is not the most efficient implementation, I wanted to make this neural network easier
to digest from a comprehensive perspective and focused on trying to point out what each thing does.

FastNet
========
There is a class included which is called "FastNet" and it represents an incredibly small, array-based neural network
implementation. It has a hard-coded topology, but is very interesting if you want to see the iterative steps that
are taken. Additionally, it's incredibly fast (compared to the recursive based approach).

Background
========
After spending an ungodly amount of time trying to make sense of the backpropagation algorithm,
something clicked for me. I wish to share this knowledge with others, hopefully saving someone a year of pondering!
