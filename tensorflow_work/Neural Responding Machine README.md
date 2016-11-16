This code implements Li Hang's "Neural Responding Machine" paper by using tensorflow framework.

Because there are too many constraints in the original RNN templates proposed by the official guide, and it is hard to do some modifications on the code. I have implemented a totally new sequence-to-sequence attention model by myself except the bucket trick. So the efficiency of MY model is slightly lower than the official version but acceptable.
