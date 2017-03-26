
## Write a trainer

The existing trainers should be enough for single-cost optimization tasks. If you
want to do something inside the trainer, consider writing it as a callback, or
write an issue to see if there is a better solution than creating new trainers.

For certain tasks, you might need a new trainer.
The [GAN trainer](../../examples/GAN/GAN.py) is one example of how to implement
new trainers.

More details to come.
