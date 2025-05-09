# Multi-Scale Pixel Similarity

Multi-scale Pixel Similarity is a simple, strict FR-IQA score used by:

https://github.com/tcz/rb-crawler

and

https://huggingface.co/datasets/tcz/rb-large

## Definition

$$  \text{MSPS}\bigl(I_1, I_2\bigr) 
  \;=\; 1 
  \;-\; \\
  \frac{1}{N} \sum_{i=1}^{N}
  \left[
    \frac{1}{H_i W_i C} 
    \sum_{h=1}^{H_i} \sum_{w=1}^{W_i} \sum_{c=1}^{C}
    \Bigl( I_1^{(i)}(h, w, c) - I_2^{(i)}(h, w, c) \Bigr)^2
  \right]$$

Here $I_1$ and $I_2$ are the input images to compare, and $I_1^{(i)}$ and $I_2^{(i)}$ are their scaled down versions. $I_1^{(1)}$ and $I_2^{(1)}$ refer to the original images without scaling, while $I_1^{(2)}$ and $I_2^{(2)}$ are half the size of the originals, etc. Resizing is done by 2D average-pooling. 

$N$ is the total number of MSE calculations and can be calculated like so:

$$ N \;=\; 1 \;+\; \left\lfloor \log_{2}\!\Bigl(\min(H_1, W_1)\Bigr) \right\rfloor $$ 
