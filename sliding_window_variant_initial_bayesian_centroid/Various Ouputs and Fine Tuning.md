**Non-fine tuned output**

\=================================================================  
🧠 BAYESIAN CENTROID MEMORY DIAGNOSTICS  
\=================================================================  
Total Prototypes:         512  
Average Variance (σ²):    1.0090  (Prior \= 1.0000)  
Min Variance (Sharpest):  0.7276  
Max Variance (Diffuse):   1.3042  
Prototype Norm Mean:      1.3403  
Active Memory Spread:     0.4375

\=================================================================  
🗺️ COGNITIVE MAP & JEPA MECHANISM PROBE  
\=================================================================  
Cognitive Map Step Velocity (Avg): 3.3783  
Total Manifold Path Displacement:   42.6796  
Centroids Activated by Prompt:      11/512 unique clusters  
JEPA Next-Latent Cosine Sim:        0.7358  (73.6% alignment)

\=================================================================  
⚡ GENERATIVE PROBES & HARDWARE THROUGHPUT BENCHMARKS  
Sampling: Temp=0.65 | Top-P=0.85 | Top-K=40 | Rep-Penalty=1.2  
\=================================================================

\--- Domain: Web/Educational \---  
The theory of general relativity explains that gravity is the force that keeps objects in space. However, it does not explain how objects can move around each other.  
To understand why people think they are “in control” with this theory we need to consider a little bit about what happens when things change over time and space. For example: If you roll your eyes from side to side on an airplane, will it still fly? The same thing happens if you turn the wheel on, or turn the handlebars – these are examples of “kinetic” objects. They cannot be considered as being in control because there is no external influence on them.  
When something changes, it is called a “molecular force.” A molecular force is a force that has a mass and is related to its magnitude (or force) and direction (or velocity). In other words, it can be described by a “force field.” The force field is what causes things to move and change over time and space.  
This theory explains why people think they are in control with this theory. However, this does not explain how the objects move around each other or if they are moving away from one another – for example, you could say that a car will accelerate as it moves across the road. This idea of motion is known as “kinetic” objects.  
For example, imagine you have a box full of toys. You want to find out which toy has the smallest amount of toys, but  
\-----------------------------------------------------------------  
  📥 Input Speed (Prefill):    1,871.7 tok/s  (9 tokens in 4.81 ms)  
  📤 Output Speed (Decode):      180.8 tok/s  (300 tokens in 1.66 s)  
  ⏱️  Latency per Token:          5.53 ms/token  
\-----------------------------------------------------------------

\--- Domain: Synthetic Reasoning \---  
Photosynthesis is crucial for life on Earth because it provides food and energy to the plants, but it also plays a significant role in determining how plants grow. This process involves chemical reactions that break down the organic material into simpler components such as sugars, fats, and proteins. One important aspect of photosynthesis is chlorophyll, which provides light for photosynthesis by absorbing light and converting it into energy through photosynthesis.

In this chapter, we will explore some fascinating aspects of photosynthesis using real-life examples from nature. We'll discuss different types of photosynthesis, their characteristics, and why they are essential for plant growth. By understanding these processes, you can appreciate the complexity of life and its interconnectedness with other organisms.

Section 1: What is Photosynthesis?  
\----------------------------------

At first glance, photosynthesis may seem like a simple process involving water and carbon dioxide. However, there's much more to it than meets the eye\! Plants rely on photosynthesis to convert sunlight into energy through photosynthesis. This unique ability allows them to survive in harsh environments such as deserts, forests, or even oceans.

One famous example of photosynthesis is the transformation of glucose (a type of sugar) from a primary source (the sun) into another form called chlorophyll. When this happens, chlorophyll absorbs sunlight and converts it into energy via chlorophyll production. As a result, plants absorb light at different wavelengths depending upon their environment – red, green, and blue – which serves as the primary  
\-----------------------------------------------------------------  
  📥 Input Speed (Prefill):    1,662.9 tok/s  (9 tokens in 5.41 ms)  
  📤 Output Speed (Decode):      195.0 tok/s  (300 tokens in 1.54 s)  
  ⏱️  Latency per Token:          5.13 ms/token  
\-----------------------------------------------------------------

\--- Domain: Python Code \---  
def quicksort(arr):  
    """Sorts an array recursively."""  
    arr \= np.zeros((len(arr), dtype=int, int\_dtype='float32'))  
    for i in range(1, len(arr)):  
        if arr\[i\] \== 1:  
            arr\[i\] \= arr

    return arr  
\`\`\`  
The \`quicksort()\` function takes two arguments: the first argument is a list of integers that define the size of each element in the array, and the second argument is an integer that specifies which elements are to be added or removed from the array. The function returns a tuple containing the values of the elements in the array.

\#\#\# Calculating Elements in a Array

Let's consider an array of integers, where each element corresponds to a row in the array and all other elements correspond to rows in the array. We can calculate the element using a simple formula:  
$$  
\\text{element} \= \\begin{cases}  
1 & \\text{if } n \> 0 \\\\  
0 & 1 & \\text{otherwise}  
\\end{cases}  
$$  
where \`n\` is the number of rows in the array.  
\`\`\`python  
print(arr\[0\])  \# Output: {'n': 0}  
\`\`\`  
The output will be \`None\`.

\#\#\# Calcul  
\-----------------------------------------------------------------  
  📥 Input Speed (Prefill):    4,543.8 tok/s  (22 tokens in 4.84 ms)  
  📤 Output Speed (Decode):      195.6 tok/s  (300 tokens in 1.53 s)  
  ⏱️  Latency per Token:          5.11 ms/token  
\-----------------------------------------------------------------

**x.1 LR for bayesian centroid – good but seemed to overfit at the end of training**

\=================================================================  
\[transformers\] Token indices sequence length is longer than the specified maximum sequence length for this model (1055 \> 1024). Running this sequence through the model will result in indexing errors  
SFT Step    25/5000 | Response PPL:   7.57 | CE: 2.0246 | JEPA: 0.4923 | LR: 6.25e-06  
SFT Step    50/5000 | Response PPL:   6.78 | CE: 1.9140 | JEPA: 0.3833 | LR: 1.25e-05  
SFT Step    75/5000 | Response PPL:   6.68 | CE: 1.8998 | JEPA: 0.3660 | LR: 1.88e-05  
SFT Step   100/5000 | Response PPL:   6.51 | CE: 1.8728 | JEPA: 0.3544 | LR: 2.50e-05  
SFT Step   125/5000 | Response PPL:   6.07 | CE: 1.8036 | JEPA: 0.3338 | LR: 3.13e-05  
SFT Step   150/5000 | Response PPL:   6.27 | CE: 1.8359 | JEPA: 0.3389 | LR: 3.75e-05  
SFT Step   175/5000 | Response PPL:   6.21 | CE: 1.8256 | JEPA: 0.3248 | LR: 4.38e-05  
SFT Step   200/5000 | Response PPL:   6.36 | CE: 1.8497 | JEPA: 0.3400 | LR: 5.00e-05  
SFT Step   225/5000 | Response PPL:   5.86 | CE: 1.7677 | JEPA: 0.3405 | LR: 5.00e-05  
SFT Step   250/5000 | Response PPL:   5.81 | CE: 1.7599 | JEPA: 0.3228 | LR: 5.00e-05  
SFT Step   275/5000 | Response PPL:   5.88 | CE: 1.7717 | JEPA: 0.3261 | LR: 5.00e-05  
SFT Step   300/5000 | Response PPL:   5.77 | CE: 1.7525 | JEPA: 0.3205 | LR: 5.00e-05  
SFT Step   325/5000 | Response PPL:   5.69 | CE: 1.7383 | JEPA: 0.3183 | LR: 4.99e-05  
SFT Step   350/5000 | Response PPL:   5.54 | CE: 1.7126 | JEPA: 0.3205 | LR: 4.99e-05  
SFT Step   375/5000 | Response PPL:   5.68 | CE: 1.7367 | JEPA: 0.3236 | LR: 4.99e-05  
SFT Step   400/5000 | Response PPL:   5.66 | CE: 1.7327 | JEPA: 0.3161 | LR: 4.98e-05  
SFT Step   425/5000 | Response PPL:   5.60 | CE: 1.7220 | JEPA: 0.3269 | LR: 4.98e-05  
SFT Step   450/5000 | Response PPL:   5.71 | CE: 1.7417 | JEPA: 0.3125 | LR: 4.97e-05  
SFT Step   475/5000 | Response PPL:   5.45 | CE: 1.6960 | JEPA: 0.3154 | LR: 4.96e-05  
SFT Step   500/5000 | Response PPL:   5.32 | CE: 1.6715 | JEPA: 0.3132 | LR: 4.96e-05  
SFT Step   525/5000 | Response PPL:   5.39 | CE: 1.6850 | JEPA: 0.3163 | LR: 4.95e-05  
SFT Step   550/5000 | Response PPL:   5.11 | CE: 1.6318 | JEPA: 0.3189 | LR: 4.94e-05  
SFT Step   575/5000 | Response PPL:   5.41 | CE: 1.6882 | JEPA: 0.3199 | LR: 4.93e-05  
SFT Step   600/5000 | Response PPL:   5.50 | CE: 1.7047 | JEPA: 0.3199 | LR: 4.92e-05  
SFT Step   625/5000 | Response PPL:   5.24 | CE: 1.6565 | JEPA: 0.3152 | LR: 4.91e-05  
SFT Step   650/5000 | Response PPL:   5.52 | CE: 1.7081 | JEPA: 0.3201 | LR: 4.90e-05  
SFT Step   675/5000 | Response PPL:   5.30 | CE: 1.6668 | JEPA: 0.3109 | LR: 4.89e-05  
SFT Step   700/5000 | Response PPL:   5.31 | CE: 1.6690 | JEPA: 0.3222 | LR: 4.88e-05  
SFT Step   725/5000 | Response PPL:   5.36 | CE: 1.6781 | JEPA: 0.3251 | LR: 4.87e-05  
SFT Step   750/5000 | Response PPL:   5.34 | CE: 1.6754 | JEPA: 0.3143 | LR: 4.86e-05  
SFT Step   775/5000 | Response PPL:   5.37 | CE: 1.6799 | JEPA: 0.3271 | LR: 4.84e-05  
SFT Step   800/5000 | Response PPL:   5.18 | CE: 1.6456 | JEPA: 0.3210 | LR: 4.83e-05  
SFT Step   825/5000 | Response PPL:   5.10 | CE: 1.6290 | JEPA: 0.3111 | LR: 4.81e-05  
SFT Step   850/5000 | Response PPL:   5.21 | CE: 1.6510 | JEPA: 0.3253 | LR: 4.80e-05  
SFT Step   875/5000 | Response PPL:   5.12 | CE: 1.6324 | JEPA: 0.3131 | LR: 4.78e-05  
SFT Step   900/5000 | Response PPL:   5.21 | CE: 1.6497 | JEPA: 0.3188 | LR: 4.77e-05  
SFT Step   925/5000 | Response PPL:   5.08 | CE: 1.6249 | JEPA: 0.3215 | LR: 4.75e-05  
SFT Step   950/5000 | Response PPL:   5.14 | CE: 1.6367 | JEPA: 0.3208 | LR: 4.74e-05  
SFT Step   975/5000 | Response PPL:   5.12 | CE: 1.6333 | JEPA: 0.3220 | LR: 4.72e-05  
SFT Step  1000/5000 | Response PPL:   5.37 | CE: 1.6804 | JEPA: 0.3224 | LR: 4.70e-05  
\--\> Saved SFT Checkpoint: ./checkpoints\_sft\\sft\_step\_1000.pt  
SFT Step  1025/5000 | Response PPL:   5.40 | CE: 1.6857 | JEPA: 0.3246 | LR: 4.68e-05  
SFT Step  1050/5000 | Response PPL:   5.21 | CE: 1.6508 | JEPA: 0.3212 | LR: 4.66e-05  
SFT Step  1075/5000 | Response PPL:   5.06 | CE: 1.6218 | JEPA: 0.3187 | LR: 4.64e-05  
SFT Step  1100/5000 | Response PPL:   5.16 | CE: 1.6413 | JEPA: 0.3280 | LR: 4.62e-05  
SFT Step  1125/5000 | Response PPL:   5.20 | CE: 1.6477 | JEPA: 0.3278 | LR: 4.60e-05  
SFT Step  1150/5000 | Response PPL:   5.39 | CE: 1.6853 | JEPA: 0.3214 | LR: 4.58e-05  
SFT Step  1175/5000 | Response PPL:   5.02 | CE: 1.6131 | JEPA: 0.3160 | LR: 4.56e-05  
SFT Step  1200/5000 | Response PPL:   5.06 | CE: 1.6215 | JEPA: 0.3212 | LR: 4.54e-05  
SFT Step  1225/5000 | Response PPL:   5.26 | CE: 1.6593 | JEPA: 0.3323 | LR: 4.51e-05  
SFT Step  1250/5000 | Response PPL:   5.12 | CE: 1.6331 | JEPA: 0.3195 | LR: 4.49e-05  
SFT Step  1275/5000 | Response PPL:   5.10 | CE: 1.6296 | JEPA: 0.3282 | LR: 4.47e-05  
SFT Step  1300/5000 | Response PPL:   5.21 | CE: 1.6514 | JEPA: 0.3226 | LR: 4.44e-05  
SFT Step  1325/5000 | Response PPL:   4.91 | CE: 1.5917 | JEPA: 0.3249 | LR: 4.42e-05  
SFT Step  1350/5000 | Response PPL:   4.95 | CE: 1.5997 | JEPA: 0.3291 | LR: 4.39e-05  
SFT Step  1375/5000 | Response PPL:   5.13 | CE: 1.6360 | JEPA: 0.3210 | LR: 4.37e-05  
SFT Step  1400/5000 | Response PPL:   5.03 | CE: 1.6156 | JEPA: 0.3201 | LR: 4.34e-05  
SFT Step  1425/5000 | Response PPL:   5.06 | CE: 1.6207 | JEPA: 0.3240 | LR: 4.32e-05  
SFT Step  1450/5000 | Response PPL:   5.22 | CE: 1.6527 | JEPA: 0.3238 | LR: 4.29e-05  
SFT Step  1475/5000 | Response PPL:   5.25 | CE: 1.6590 | JEPA: 0.3209 | LR: 4.26e-05  
SFT Step  1500/5000 | Response PPL:   5.14 | CE: 1.6371 | JEPA: 0.3285 | LR: 4.23e-05  
SFT Step  1525/5000 | Response PPL:   5.03 | CE: 1.6159 | JEPA: 0.3225 | LR: 4.21e-05  
SFT Step  1550/5000 | Response PPL:   5.03 | CE: 1.6161 | JEPA: 0.3226 | LR: 4.18e-05  
SFT Step  1575/5000 | Response PPL:   5.27 | CE: 1.6620 | JEPA: 0.3362 | LR: 4.15e-05  
SFT Step  1600/5000 | Response PPL:   5.06 | CE: 1.6206 | JEPA: 0.3194 | LR: 4.12e-05  
SFT Step  1625/5000 | Response PPL:   5.09 | CE: 1.6277 | JEPA: 0.3238 | LR: 4.09e-05  
SFT Step  1650/5000 | Response PPL:   5.06 | CE: 1.6212 | JEPA: 0.3166 | LR: 4.06e-05  
SFT Step  1675/5000 | Response PPL:   4.87 | CE: 1.5834 | JEPA: 0.3195 | LR: 4.03e-05  
SFT Step  1700/5000 | Response PPL:   4.95 | CE: 1.5993 | JEPA: 0.3106 | LR: 4.00e-05  
SFT Step  1725/5000 | Response PPL:   5.13 | CE: 1.6355 | JEPA: 0.3211 | LR: 3.97e-05  
SFT Step  1750/5000 | Response PPL:   4.99 | CE: 1.6080 | JEPA: 0.3190 | LR: 3.94e-05  
SFT Step  1775/5000 | Response PPL:   5.01 | CE: 1.6112 | JEPA: 0.3205 | LR: 3.91e-05  
SFT Step  1800/5000 | Response PPL:   5.02 | CE: 1.6137 | JEPA: 0.3340 | LR: 3.88e-05  
SFT Step  1825/5000 | Response PPL:   4.84 | CE: 1.5766 | JEPA: 0.3267 | LR: 3.84e-05  
SFT Step  1850/5000 | Response PPL:   4.95 | CE: 1.5987 | JEPA: 0.3296 | LR: 3.81e-05  
SFT Step  1875/5000 | Response PPL:   5.01 | CE: 1.6124 | JEPA: 0.3334 | LR: 3.78e-05  
SFT Step  1900/5000 | Response PPL:   4.95 | CE: 1.5998 | JEPA: 0.3197 | LR: 3.75e-05  
SFT Step  1925/5000 | Response PPL:   5.14 | CE: 1.6370 | JEPA: 0.3265 | LR: 3.71e-05  
SFT Step  1950/5000 | Response PPL:   5.01 | CE: 1.6124 | JEPA: 0.3285 | LR: 3.68e-05  
SFT Step  1975/5000 | Response PPL:   5.09 | CE: 1.6281 | JEPA: 0.3386 | LR: 3.65e-05  
SFT Step  2000/5000 | Response PPL:   4.87 | CE: 1.5830 | JEPA: 0.3187 | LR: 3.61e-05  
\--\> Saved SFT Checkpoint: ./checkpoints\_sft\\sft\_step\_2000.pt  
SFT Step  2025/5000 | Response PPL:   5.11 | CE: 1.6314 | JEPA: 0.3218 | LR: 3.58e-05  
SFT Step  2050/5000 | Response PPL:   4.83 | CE: 1.5757 | JEPA: 0.3317 | LR: 3.54e-05  
SFT Step  2075/5000 | Response PPL:   5.04 | CE: 1.6178 | JEPA: 0.3242 | LR: 3.51e-05  
SFT Step  2100/5000 | Response PPL:   4.90 | CE: 1.5883 | JEPA: 0.3243 | LR: 3.47e-05  
SFT Step  2125/5000 | Response PPL:   4.96 | CE: 1.6021 | JEPA: 0.3220 | LR: 3.44e-05  
SFT Step  2150/5000 | Response PPL:   5.00 | CE: 1.6094 | JEPA: 0.3309 | LR: 3.40e-05  
SFT Step  2175/5000 | Response PPL:   4.99 | CE: 1.6080 | JEPA: 0.3360 | LR: 3.37e-05  
SFT Step  2200/5000 | Response PPL:   4.74 | CE: 1.5569 | JEPA: 0.3145 | LR: 3.33e-05  
SFT Step  2225/5000 | Response PPL:   4.99 | CE: 1.6065 | JEPA: 0.3317 | LR: 3.30e-05  
SFT Step  2250/5000 | Response PPL:   4.93 | CE: 1.5963 | JEPA: 0.3145 | LR: 3.26e-05  
SFT Step  2275/5000 | Response PPL:   4.97 | CE: 1.6037 | JEPA: 0.3255 | LR: 3.23e-05  
SFT Step  2300/5000 | Response PPL:   4.93 | CE: 1.5963 | JEPA: 0.3264 | LR: 3.19e-05  
SFT Step  2325/5000 | Response PPL:   4.93 | CE: 1.5960 | JEPA: 0.3275 | LR: 3.15e-05  
SFT Step  2350/5000 | Response PPL:   5.02 | CE: 1.6131 | JEPA: 0.3283 | LR: 3.12e-05  
SFT Step  2375/5000 | Response PPL:   5.07 | CE: 1.6231 | JEPA: 0.3260 | LR: 3.08e-05  
SFT Step  2400/5000 | Response PPL:   4.95 | CE: 1.6002 | JEPA: 0.3267 | LR: 3.05e-05  
SFT Step  2425/5000 | Response PPL:   4.91 | CE: 1.5918 | JEPA: 0.3233 | LR: 3.01e-05  
SFT Step  2450/5000 | Response PPL:   5.02 | CE: 1.6136 | JEPA: 0.3248 | LR: 2.97e-05  
SFT Step  2475/5000 | Response PPL:   4.81 | CE: 1.5707 | JEPA: 0.3245 | LR: 2.94e-05  
SFT Step  2500/5000 | Response PPL:   5.22 | CE: 1.6516 | JEPA: 0.3330 | LR: 2.90e-05  
SFT Step  2525/5000 | Response PPL:   4.81 | CE: 1.5701 | JEPA: 0.3277 | LR: 2.86e-05  
SFT Step  2550/5000 | Response PPL:   4.90 | CE: 1.5886 | JEPA: 0.3275 | LR: 2.83e-05  
SFT Step  2575/5000 | Response PPL:   4.84 | CE: 1.5774 | JEPA: 0.3362 | LR: 2.79e-05  
SFT Step  2600/5000 | Response PPL:   4.89 | CE: 1.5871 | JEPA: 0.3208 | LR: 2.75e-05  
SFT Step  2625/5000 | Response PPL:   4.95 | CE: 1.6001 | JEPA: 0.3321 | LR: 2.71e-05  
SFT Step  2650/5000 | Response PPL:   4.92 | CE: 1.5942 | JEPA: 0.3186 | LR: 2.68e-05  
SFT Step  2675/5000 | Response PPL:   4.71 | CE: 1.5486 | JEPA: 0.3201 | LR: 2.64e-05  
SFT Step  2700/5000 | Response PPL:   4.80 | CE: 1.5693 | JEPA: 0.3153 | LR: 2.60e-05  
SFT Step  2725/5000 | Response PPL:   4.93 | CE: 1.5961 | JEPA: 0.3358 | LR: 2.57e-05  
SFT Step  2750/5000 | Response PPL:   4.99 | CE: 1.6070 | JEPA: 0.3362 | LR: 2.53e-05  
SFT Step  2775/5000 | Response PPL:   4.74 | CE: 1.5560 | JEPA: 0.3215 | LR: 2.49e-05  
SFT Step  2800/5000 | Response PPL:   4.97 | CE: 1.6043 | JEPA: 0.3298 | LR: 2.46e-05  
SFT Step  2825/5000 | Response PPL:   4.77 | CE: 1.5615 | JEPA: 0.3361 | LR: 2.42e-05  
SFT Step  2850/5000 | Response PPL:   4.75 | CE: 1.5583 | JEPA: 0.3160 | LR: 2.38e-05  
SFT Step  2875/5000 | Response PPL:   4.86 | CE: 1.5818 | JEPA: 0.3180 | LR: 2.35e-05  
SFT Step  2900/5000 | Response PPL:   4.91 | CE: 1.5921 | JEPA: 0.3249 | LR: 2.31e-05  
SFT Step  2925/5000 | Response PPL:   4.78 | CE: 1.5651 | JEPA: 0.3243 | LR: 2.28e-05  
SFT Step  2950/5000 | Response PPL:   4.98 | CE: 1.6051 | JEPA: 0.3253 | LR: 2.24e-05  
SFT Step  2975/5000 | Response PPL:   4.79 | CE: 1.5660 | JEPA: 0.3204 | LR: 2.20e-05  
SFT Step  3000/5000 | Response PPL:   4.97 | CE: 1.6026 | JEPA: 0.3250 | LR: 2.17e-05  
\--\> Saved SFT Checkpoint: ./checkpoints\_sft\\sft\_step\_3000.pt  
SFT Step  3025/5000 | Response PPL:   4.83 | CE: 1.5757 | JEPA: 0.3239 | LR: 2.13e-05  
SFT Step  3050/5000 | Response PPL:   4.94 | CE: 1.5970 | JEPA: 0.3292 | LR: 2.10e-05  
SFT Step  3075/5000 | Response PPL:   4.83 | CE: 1.5743 | JEPA: 0.3263 | LR: 2.06e-05  
SFT Step  3100/5000 | Response PPL:   5.04 | CE: 1.6180 | JEPA: 0.3360 | LR: 2.03e-05  
SFT Step  3125/5000 | Response PPL:   4.90 | CE: 1.5884 | JEPA: 0.3364 | LR: 1.99e-05  
SFT Step  3150/5000 | Response PPL:   4.86 | CE: 1.5819 | JEPA: 0.3249 | LR: 1.96e-05  
SFT Step  3175/5000 | Response PPL:   4.87 | CE: 1.5821 | JEPA: 0.3279 | LR: 1.92e-05  
SFT Step  3200/5000 | Response PPL:   4.66 | CE: 1.5395 | JEPA: 0.3187 | LR: 1.89e-05  
SFT Step  3225/5000 | Response PPL:   4.76 | CE: 1.5592 | JEPA: 0.3292 | LR: 1.86e-05  
SFT Step  3250/5000 | Response PPL:   5.01 | CE: 1.6112 | JEPA: 0.3324 | LR: 1.82e-05  
SFT Step  3275/5000 | Response PPL:   5.11 | CE: 1.6309 | JEPA: 0.3360 | LR: 1.79e-05  
SFT Step  3300/5000 | Response PPL:   4.72 | CE: 1.5514 | JEPA: 0.3246 | LR: 1.76e-05  
SFT Step  3325/5000 | Response PPL:   4.77 | CE: 1.5633 | JEPA: 0.3302 | LR: 1.72e-05  
SFT Step  3350/5000 | Response PPL:   4.63 | CE: 1.5328 | JEPA: 0.3245 | LR: 1.69e-05  
SFT Step  3375/5000 | Response PPL:   4.72 | CE: 1.5528 | JEPA: 0.3290 | LR: 1.66e-05  
SFT Step  3400/5000 | Response PPL:   4.74 | CE: 1.5555 | JEPA: 0.3253 | LR: 1.63e-05  
SFT Step  3425/5000 | Response PPL:   4.66 | CE: 1.5389 | JEPA: 0.3149 | LR: 1.59e-05  
SFT Step  3450/5000 | Response PPL:   4.76 | CE: 1.5612 | JEPA: 0.3103 | LR: 1.56e-05  
SFT Step  3475/5000 | Response PPL:   4.98 | CE: 1.6062 | JEPA: 0.3283 | LR: 1.53e-05  
SFT Step  3500/5000 | Response PPL:   4.75 | CE: 1.5589 | JEPA: 0.3256 | LR: 1.50e-05  
SFT Step  3525/5000 | Response PPL:   4.77 | CE: 1.5626 | JEPA: 0.3328 | LR: 1.47e-05  
SFT Step  3550/5000 | Response PPL:   4.79 | CE: 1.5671 | JEPA: 0.3248 | LR: 1.44e-05  
SFT Step  3575/5000 | Response PPL:   4.84 | CE: 1.5760 | JEPA: 0.3227 | LR: 1.41e-05  
SFT Step  3600/5000 | Response PPL:   4.83 | CE: 1.5743 | JEPA: 0.3206 | LR: 1.38e-05  
SFT Step  3625/5000 | Response PPL:   4.72 | CE: 1.5512 | JEPA: 0.3278 | LR: 1.35e-05  
SFT Step  3650/5000 | Response PPL:   4.96 | CE: 1.6014 | JEPA: 0.3290 | LR: 1.32e-05  
SFT Step  3675/5000 | Response PPL:   4.79 | CE: 1.5660 | JEPA: 0.3148 | LR: 1.30e-05  
SFT Step  3700/5000 | Response PPL:   5.07 | CE: 1.6227 | JEPA: 0.3258 | LR: 1.27e-05  
SFT Step  3725/5000 | Response PPL:   4.68 | CE: 1.5439 | JEPA: 0.3266 | LR: 1.24e-05  
SFT Step  3750/5000 | Response PPL:   4.97 | CE: 1.6043 | JEPA: 0.3347 | LR: 1.21e-05  
SFT Step  3775/5000 | Response PPL:   4.98 | CE: 1.6050 | JEPA: 0.3290 | LR: 1.19e-05  
SFT Step  3800/5000 | Response PPL:   4.58 | CE: 1.5215 | JEPA: 0.3155 | LR: 1.16e-05  
SFT Step  3825/5000 | Response PPL:   4.58 | CE: 1.5208 | JEPA: 0.3206 | LR: 1.13e-05  
SFT Step  3850/5000 | Response PPL:   4.89 | CE: 1.5867 | JEPA: 0.3351 | LR: 1.11e-05  
SFT Step  3875/5000 | Response PPL:   4.62 | CE: 1.5305 | JEPA: 0.3267 | LR: 1.08e-05  
SFT Step  3900/5000 | Response PPL:   4.79 | CE: 1.5668 | JEPA: 0.3335 | LR: 1.06e-05  
SFT Step  3925/5000 | Response PPL:   4.79 | CE: 1.5657 | JEPA: 0.3264 | LR: 1.04e-05  
SFT Step  3950/5000 | Response PPL:   4.74 | CE: 1.5571 | JEPA: 0.3416 | LR: 1.01e-05  
SFT Step  3975/5000 | Response PPL:   4.78 | CE: 1.5650 | JEPA: 0.3234 | LR: 9.89e-06  
SFT Step  4000/5000 | Response PPL:   4.80 | CE: 1.5687 | JEPA: 0.3396 | LR: 9.66e-06  
\--\> Saved SFT Checkpoint: ./checkpoints\_sft\\sft\_step\_4000.pt  
SFT Step  4025/5000 | Response PPL:   4.89 | CE: 1.5878 | JEPA: 0.3355 | LR: 9.44e-06  
SFT Step  4050/5000 | Response PPL:   4.72 | CE: 1.5515 | JEPA: 0.3179 | LR: 9.22e-06  
SFT Step  4075/5000 | Response PPL:   4.79 | CE: 1.5673 | JEPA: 0.3291 | LR: 9.01e-06  
SFT Step  4100/5000 | Response PPL:   4.78 | CE: 1.5652 | JEPA: 0.3236 | LR: 8.80e-06  
SFT Step  4125/5000 | Response PPL:   4.68 | CE: 1.5441 | JEPA: 0.3304 | LR: 8.60e-06  
SFT Step  4150/5000 | Response PPL:   4.78 | CE: 1.5643 | JEPA: 0.3248 | LR: 8.40e-06  
SFT Step  4175/5000 | Response PPL:   4.77 | CE: 1.5628 | JEPA: 0.3310 | LR: 8.21e-06  
SFT Step  4200/5000 | Response PPL:   4.64 | CE: 1.5354 | JEPA: 0.3261 | LR: 8.02e-06  
SFT Step  4225/5000 | Response PPL:   4.74 | CE: 1.5562 | JEPA: 0.3310 | LR: 7.84e-06  
SFT Step  4250/5000 | Response PPL:   4.79 | CE: 1.5655 | JEPA: 0.3182 | LR: 7.66e-06  
SFT Step  4275/5000 | Response PPL:   4.84 | CE: 1.5778 | JEPA: 0.3251 | LR: 7.49e-06  
SFT Step  4300/5000 | Response PPL:   4.70 | CE: 1.5480 | JEPA: 0.3226 | LR: 7.33e-06  
SFT Step  4325/5000 | Response PPL:   4.73 | CE: 1.5537 | JEPA: 0.3187 | LR: 7.17e-06  
SFT Step  4350/5000 | Response PPL:   4.73 | CE: 1.5548 | JEPA: 0.3216 | LR: 7.01e-06  
SFT Step  4375/5000 | Response PPL:   4.69 | CE: 1.5455 | JEPA: 0.3243 | LR: 6.86e-06  
SFT Step  4400/5000 | Response PPL:   4.86 | CE: 1.5808 | JEPA: 0.3246 | LR: 6.72e-06  
SFT Step  4425/5000 | Response PPL:   4.75 | CE: 1.5571 | JEPA: 0.3083 | LR: 6.58e-06  
SFT Step  4450/5000 | Response PPL:   4.64 | CE: 1.5337 | JEPA: 0.3201 | LR: 6.45e-06  
SFT Step  4475/5000 | Response PPL:   4.62 | CE: 1.5303 | JEPA: 0.3215 | LR: 6.32e-06  
SFT Step  4500/5000 | Response PPL:   4.82 | CE: 1.5736 | JEPA: 0.3247 | LR: 6.20e-06  
SFT Step  4525/5000 | Response PPL:   4.70 | CE: 1.5467 | JEPA: 0.3218 | LR: 6.08e-06  
SFT Step  4550/5000 | Response PPL:   4.72 | CE: 1.5515 | JEPA: 0.3364 | LR: 5.97e-06  
SFT Step  4575/5000 | Response PPL:   4.70 | CE: 1.5466 | JEPA: 0.3283 | LR: 5.87e-06  
SFT Step  4600/5000 | Response PPL:   4.68 | CE: 1.5440 | JEPA: 0.3207 | LR: 5.77e-06  
SFT Step  4625/5000 | Response PPL:   4.64 | CE: 1.5344 | JEPA: 0.3187 | LR: 5.68e-06  
SFT Step  4650/5000 | Response PPL:   4.74 | CE: 1.5569 | JEPA: 0.3209 | LR: 5.59e-06  
SFT Step  4675/5000 | Response PPL:   4.60 | CE: 1.5270 | JEPA: 0.3166 | LR: 5.51e-06  
SFT Step  4700/5000 | Response PPL:   4.80 | CE: 1.5688 | JEPA: 0.3214 | LR: 5.44e-06  
SFT Step  4725/5000 | Response PPL:   4.82 | CE: 1.5726 | JEPA: 0.3315 | LR: 5.37e-06  
SFT Step  4750/5000 | Response PPL:   4.65 | CE: 1.5379 | JEPA: 0.3239 | LR: 5.30e-06  
SFT Step  4775/5000 | Response PPL:   4.59 | CE: 1.5236 | JEPA: 0.3229 | LR: 5.25e-06  
SFT Step  4800/5000 | Response PPL:   4.80 | CE: 1.5677 | JEPA: 0.3245 | LR: 5.19e-06  
SFT Step  4825/5000 | Response PPL:   4.70 | CE: 1.5484 | JEPA: 0.3228 | LR: 5.15e-06  
SFT Step  4850/5000 | Response PPL:   4.65 | CE: 1.5371 | JEPA: 0.3179 | LR: 5.11e-06  
SFT Step  4875/5000 | Response PPL:   4.81 | CE: 1.5707 | JEPA: 0.3330 | LR: 5.08e-06  
SFT Step  4900/5000 | Response PPL:   4.95 | CE: 1.5988 | JEPA: 0.3285 | LR: 5.05e-06  
SFT Step  4925/5000 | Response PPL:   4.62 | CE: 1.5311 | JEPA: 0.3310 | LR: 5.03e-06  
SFT Step  4950/5000 | Response PPL:   4.61 | CE: 1.5284 | JEPA: 0.3253 | LR: 5.01e-06  
SFT Step  4975/5000 | Response PPL:   4.83 | CE: 1.5747 | JEPA: 0.3341 | LR: 5.00e-06  
SFT Step  5000/5000 | Response PPL:   4.85 | CE: 1.5795 | JEPA: 0.3320 | LR: 5.00e-06  
\--\> Saved SFT Checkpoint: ./checkpoints\_sft\\sft\_step\_5000.pt

**1000 steps fine tuning checkpoint (tok/s is off because I was training it while testing) x .1 bayesian centroid LR**

PS C:\\SNN-JEPA\> py .\\probe\_checkpoint\_2.0.py  
✅ Successfully loaded model from Step 1000\!

\=================================================================  
🧠 BAYESIAN CENTROID MEMORY DIAGNOSTICS  
\=================================================================  
Total Prototypes:         512  
Average Variance (σ²):    1.0089  (Prior \= 1.0000)  
Min Variance (Sharpest):  0.7277  
Max Variance (Diffuse):   1.3038  
Prototype Norm Mean:      1.3407  
Active Memory Spread:     0.4375

\=================================================================  
🗺️ COGNITIVE MAP & JEPA MECHANISM PROBE  
\=================================================================  
Cognitive Map Step Velocity (Avg): 3.4362  
Total Manifold Path Displacement:   43.4317  
Centroids Activated by Prompt:      9/512 unique clusters  
JEPA Next-Latent Cosine Sim:        0.6570  (65.7% alignment)

\=================================================================  
⚡ GENERATIVE PROBES & HARDWARE THROUGHPUT BENCHMARKS  
Sampling: Temp=0.65 | Top-P=0.85 | Top-K=40 | Rep-Penalty=1.2  
\=================================================================

\--- Domain: Web/Educational \---  
The theory of general relativity explains that gravity is the force responsible for the formation of objects in space.  
In fact, it's not exactly a matter of theoretical interest, but rather an important aspect of our understanding of physics. The theory of general relativity has far-reaching implications beyond classical physics and continues to influence scientific research today.\<|im\_end|\>

One area where the theory of general relativity can be applied is in the study of gravitational waves, which are waves of massive gravitational forces that appear in the universe at the speed of light. These waves exhibit unique properties that make them difficult to detect using traditional means of measurement, such as absolute zero or zero gravity. However, they have also become increasingly useful for modeling complex systems, including the formation of galaxies and other celestial bodies.

Another key application of the theory of general relativity lies in the development of new technologies for measuring and analyzing gravitational waves. For example, the theory of relativity predicts the existence of gravitational waves, which provide a way to measure distances and velocities in the universe. The theory of general relativity has also been used to study the behavior of massive objects, such as stars, planets, and other celestial bodies.

The connection between the theory of general relativity and modern scientific research is a testament to the profound impact that mathematical models can have on our understanding of the universe. By exploring the principles underlying these concepts, we can gain new insights into the nature of reality, the universe's intricate structure, and the fundamental laws governing it.\<|im\_  
\-----------------------------------------------------------------  
  📥 Input Speed (Prefill):      211.8 tok/s  (9 tokens in 42.50 ms)  
  📤 Output Speed (Decode):       21.7 tok/s  (300 tokens in 13.80 s)  
  ⏱️  Latency per Token:         45.99 ms/token  
\-----------------------------------------------------------------

\--- Domain: Synthetic Reasoning \---  
Photosynthesis is crucial for life on Earth because it allows plants to harness light energy and create energy through photosynthesis. This process involves the transfer of water from one part of a plant to another, where sunlight, carbon dioxide, and water combine in specific patterns to produce glucose, oxygen, and other compounds that are essential for life as we know it today.\<|im\_end|\>

The process of photosynthesis begins with the breakdown of carbon dioxide into glucose and oxygen. During this stage, the plants use their energy to convert these compounds into simpler forms such as sugars or fats. As they break down the glucose molecules into glucose and oxygen, they release energy stored in the form of ATP (adenosine triphosphate) or adenosine triphosphate (ATP). This energy is then used by the plant to produce carbohydrates, proteins, and fats, which can be used for various purposes like fuel, repair, and even growth.

Once the plants have absorbed all of this energy, they start using it to power various cellular processes, including respiration, photosynthesis, and respiration. The process begins with the breakdown of carbon dioxide and water, releasing energy in the form of ATP. This energy is then released as a product of glucose and oxygen, which are essential for maintaining life on Earth.

In addition to these processes, there are other crucial steps that occur during photosynthesis that play a vital role in sustaining life on Earth: photosynthesis. During this stage, certain types of bacteria break  
\-----------------------------------------------------------------  
  📥 Input Speed (Prefill):    1,696.3 tok/s  (9 tokens in 5.31 ms)  
  📤 Output Speed (Decode):       84.0 tok/s  (300 tokens in 3.57 s)  
  ⏱️  Latency per Token:         11.90 ms/token  
\-----------------------------------------------------------------

\--- Domain: Python Code \---  
def quicksort(arr):  
    """Sorts an array recursively."""  
    for i in range(1, len(arr) \- 1):  
        if arr\[i\] \== 0:  
            return \[x \*\* i \+ y\*\* for x, y in arr\[i\]\]  
    else:  
          raise ValueError("Invalid array type")

    return 0  
\`\`\`

This implementation uses a list comprehension to iterate over the elements of the input array. It checks each element at the given index and returns it as an integer. The function then calls itself on the given array, checks its length, and returns it as an integer.\<|im\_end|\>

You can use this function like this:

\`\`\`python  
arr \= \[1, 2, 3\]  
print(f"The first element is {i}")  
\`\`\`

This will print out all elements of the original array. If there are any elements in the original array that do not match, the function returns 0\. Otherwise, it prints 1 for the specified number of elements to avoid division by zero.

Here's how you could implement this using a list comprehension:

\`\`\`python  
def main():  
    arr \= \[\]  
    for i in range(1, len(arr) \- 1):

\-----------------------------------------------------------------  
  📥 Input Speed (Prefill):    5,093.3 tok/s  (22 tokens in 4.32 ms)  
  📤 Output Speed (Decode):       61.4 tok/s  (300 tokens in 4.88 s)  
  ⏱️  Latency per Token:         16.28 ms/token  
\-----------------------------------------------------------------

2000 point checkpoint fine tuning

PS C:\\SNN-JEPA\> py .\\probe\_checkpoint\_2.0.py  
✅ Successfully loaded model from Step 2000\!

\=================================================================  
🧠 BAYESIAN CENTROID MEMORY DIAGNOSTICS  
\=================================================================  
Total Prototypes:         512  
Average Variance (σ²):    1.0089  (Prior \= 1.0000)  
Min Variance (Sharpest):  0.7279  
Max Variance (Diffuse):   1.3037  
Prototype Norm Mean:      1.3413  
Active Memory Spread:     0.4376

\=================================================================  
🗺️ COGNITIVE MAP & JEPA MECHANISM PROBE  
\=================================================================  
Cognitive Map Step Velocity (Avg): 3.2387  
Total Manifold Path Displacement:   40.3275  
Centroids Activated by Prompt:      8/512 unique clusters  
JEPA Next-Latent Cosine Sim:        0.6359  (63.6% alignment)

\=================================================================  
⚡ GENERATIVE PROBES & HARDWARE THROUGHPUT BENCHMARKS  
Sampling: Temp=0.65 | Top-P=0.85 | Top-K=40 | Rep-Penalty=1.2  
\=================================================================

\--- Domain: Web/Educational \---  
The theory of general relativity explains that gravity is not just a force acting on objects in space, but rather a result of the way we can move about the universe.

In fact, it's been shown that our understanding of gravity is limited by our own knowledge of the physical laws governing motion and mass distribution around objects. However, this does not mean that we should be able to directly observe these events or understand them through a direct method. The problem lies in the nature of gravity itself, which is an illusion created by our intuitive sense of balance between gravity and time.

One approach to solving this problem involves using a combination of quantum mechanics and physics to create a mathematical framework for describing the behavior of gravity and its effects on objects at different scales. This framework allows us to account for the symmetries of spacetime, which are the fundamental units of reality.

A key aspect of this theory is that it's based on the concept of "time" – the smallest unit of time in space, such as seconds or hours. However, time can be measured using physical laws like Planck's constant, which describes the speed of light in the universe. The fabric of space around these objects is also curved, with mass and energy interacting at speeds that are proportional to the distance between them.

This model provides a mathematical foundation for understanding gravity, but its limitations and potential applications are significant. For example, Einstein's theory of general relativity has been used to predict the behavior of objects at different scales, making it difficult to  
\-----------------------------------------------------------------  
  📥 Input Speed (Prefill):      284.7 tok/s  (9 tokens in 31.61 ms)  
  📤 Output Speed (Decode):       27.3 tok/s  (300 tokens in 10.99 s)  
  ⏱️  Latency per Token:         36.65 ms/token  
\-----------------------------------------------------------------

\--- Domain: Synthetic Reasoning \---  
Photosynthesis is crucial for life on Earth because it provides energy, oxygen, and nutrients to plants. This process also allows us to harness the power of sunlight, which can be used to convert carbon dioxide from the air into glucose, a vital nutrient that fuels our cells.\<|im\_end|\>  
The sun's energy source is made up of three main components: water, oxygen, and carbon dioxide. Water is essential for plant growth, while oxygen plays a crucial role in photosynthesis. The atmosphere is composed of these elements, with a thin layer of clouds and snow cover. It is also home to the plants that produce the oxygen we breathe, which is a key component in life on Earth.

In addition to its primary function, the sun has two main parts: the corona (the outermost part) and the coronal mass ejection (CME). Coronal mass ejection occurs when an object collides with the Sun, causing it to merge with other objects or create new ones. CMEs are formed by the interaction of the solar wind, coronal holes, and supernovae, which are the building blocks of the Sun's energy.

The process of photosynthesis is complex and involves several processes including light-dependent reactions that occur within the plant, water molecules, and the atmosphere. The sunlight provides the energy for this process through the conversion of chemical energy into chemical energy. This energy can be used to produce glucose, a vital nutrient in plants, as well as oxygen, which is essential  
\-----------------------------------------------------------------  
  📥 Input Speed (Prefill):    1,631.2 tok/s  (9 tokens in 5.52 ms)  
  📤 Output Speed (Decode):       93.4 tok/s  (300 tokens in 3.21 s)  
  ⏱️  Latency per Token:         10.71 ms/token  
\-----------------------------------------------------------------

\--- Domain: Python Code \---  
def quicksort(arr):  
    """Sorts an array recursively."""  
    if arr\[0\] \== 0:  
        return \[x for x in arr.items() if arr\[0\] \< 0 else arr\[1\].to\_index}

    arr \= \[\]  
    for i, arr in enumerate(arr):  
        if arr\[i\]\[0\] \== arr\[0\]:  
            arr\[i\]\[0\] \= arr\[i\]\[0\] \+ arr\[i\]\[0\]  
        elif arr\[i\]\[0\] \== 0:  
            return \[x for x in arr.items() if arr\[i\]\[0\] \< arr\[1\].to\_index}

    return \[arr)  
\`\`\`\<|im\_end|\>

This code uses a nested list comprehension to iterate through the array, creating two lists with \`arr\`. It then loops over each element in the list and creates an empty list called \`arr\`, which contains only the elements that are already in the current index. The loop continues until all elements have been added or removed from the list.

If you want to handle a more complex scenario like this one, you can modify it using the \`next()\` method:

\`\`\`python  
def arr\[0\]:  
\-----------------------------------------------------------------  
  📥 Input Speed (Prefill):    4,977.9 tok/s  (22 tokens in 4.42 ms)  
  📤 Output Speed (Decode):       64.3 tok/s  (300 tokens in 4.66 s)  
  ⏱️  Latency per Token:         15.54 ms/token  
\-----------------------------------------------------------------

PS C:\\SNN-JEPA\> py .\\probe\_checkpoint\_2.0.py

✅ Successfully loaded model from Step 5000\!

\=================================================================  
🧠 BAYESIAN CENTROID MEMORY DIAGNOSTICS  
\=================================================================  
Total Prototypes:         512  
Average Variance (σ²):    1.0090  (Prior \= 1.0000)  
Min Variance (Sharpest):  0.7286  
Max Variance (Diffuse):   1.3035  
Prototype Norm Mean:      1.3416  
Active Memory Spread:     0.4378

\=================================================================  
🗺️ COGNITIVE MAP & JEPA MECHANISM PROBE  
\=================================================================  
Cognitive Map Step Velocity (Avg): 3.1783  
Total Manifold Path Displacement:   39.2900  
Centroids Activated by Prompt:      9/512 unique clusters  
JEPA Next-Latent Cosine Sim:        0.6340  (63.4% alignment)

\=================================================================  
⚡ GENERATIVE PROBES & HARDWARE THROUGHPUT BENCHMARKS  
Sampling: Temp=0.65 | Top-P=0.85 | Top-K=40 | Rep-Penalty=1.2  
\=================================================================

\--- Domain: Web/Educational \---  
The theory of general relativity explains that gravity is not a force but rather an curvature of spacetime. The problem asks for the existence of a gravitational field, which is the fundamental law of physics, and the answer can be found by considering two scenarios: one that describes the universe as being "vibrated" (the other is "narrowed") and the other is described as "bounded."

One possible scenario is that the universe has a strong gravitational pull on the Earth, but it does not have any gravity at all, so we cannot conclude that the universe has a strong gravitational field. This means that there must exist a physical entity that is "mixed" with the gravitational field, but its presence or absence would not necessarily mean the universe is "mixed." In this case, the existence of the gravitational field would require the existence of a force that could be felt by all objects in the universe.\<|im\_end|\>

Another possibility is that the universe's gravitational field is indeed a weakly bound, meaning it cannot exist as a continuous force between two points. However, if the gravitational field were to exist, then it would likely have a more stable form, and the existence of such a force would need to occur at some point (or one more) where no gravitational forces are present. This would involve a process called "uniform" or "constant" gravity, which could potentially produce an infinite amount of energy, but not enough to create a strong gravitational field.

It  
\-----------------------------------------------------------------  
  📥 Input Speed (Prefill):    1,657.2 tok/s  (9 tokens in 5.43 ms)  
  📤 Output Speed (Decode):      178.2 tok/s  (300 tokens in 1.68 s)  
  ⏱️  Latency per Token:          5.61 ms/token  
\-----------------------------------------------------------------

\--- Domain: Synthetic Reasoning \---  
Photosynthesis is crucial for life on Earth because it provides the energy needed to convert light into chemical energy.

The process of photosynthesis is a complex and fascinating one, with many intricacies that require careful consideration of various factors. The most important factor in determining the rate at which this process occurs is the amount of sunlight available from the sun. This can be achieved by using solar radiation as an energy source or by harnessing the energy released during photosynthesis. However, there are also limitations to consider:

\* Solar radiation has a high rate of conversion due to its low intensity compared to the sunlight we receive today, making it difficult to capture and utilize these rays for energy production.  
\* Solar radiation is not efficient enough to absorb all forms of light equally well, leading to inefficient use of the energy resources such as sunlight and water.  
\* Solar cells have limited capacity for capturing and utilizing the energy they produce, which may limit their ability to operate efficiently in space.

Additionally, the sun's position near the Earth's surface can significantly impact the rate at which the process takes place. For instance, if the sun rises over the horizon, the amount of solar energy available would be greater than if it were to rise above the surface, resulting in a higher overall rate of conversion.

To maximize efficiency in solar energy production, it is essential to consider the factors that contribute to its efficiency. One key factor is sunlight intensity: higher-intensity sunlight (e.g., direct sunlight) can increase the efficiency of photos  
\-----------------------------------------------------------------  
  📥 Input Speed (Prefill):    1,975.5 tok/s  (9 tokens in 4.56 ms)  
  📤 Output Speed (Decode):      187.4 tok/s  (300 tokens in 1.60 s)  
  ⏱️  Latency per Token:          5.34 ms/token  
\-----------------------------------------------------------------

\--- Domain: Python Code \---  
def quicksort(arr):  
    """Sorts an array recursively."""  
    arr\[0\].append((i, j) for i in range(1, len(arr))\])

    return \[num\*\*2 for num in arr if arr\[0\] \!= 0\]  
\`\`\`\<|im\_end|\>

This function works by iterating through the array and checking each element using a \`for\` loop. For each element, it checks if its index is less than or equal to the current element. If it is, then it adds one more element from the current element to the result set. Finally, it returns the resulting array.

Here's how you could implement this:

\`\`\`python  
def max\_quicksort(arr):  
    """Sorts an array recursively."""  
    arr \= \[i for i in range(1, len(arr))\]

    while True:  
        for i in range(1, len(arr)):  
            if arr\[i\]\[j\] \!= 0 or arr\[i\]\[j\] \== 1:  
                return arr\[i\]\[j\]  
            arr\[i\]\[j\] \= (arr\[i\]\[j\]) / 2  
    return arr  
\`\`  
\-----------------------------------------------------------------  
  📥 Input Speed (Prefill):    5,304.4 tok/s  (22 tokens in 4.15 ms)  
  📤 Output Speed (Decode):      209.8 tok/s  (300 tokens in 1.43 s)  
  ⏱️  Latency per Token:          4.77 ms/token  
\-----------------------------------------------------------------

\[transformers\] Token indices sequence length is longer than the specified maximum sequence length for this model (1055 \> 1024). Running this sequence through the model will result in indexing errors  
SFT Step    25/5000 | Response PPL:   7.57 | CE: 2.0246 | JEPA: 0.4923 | LR: 6.25e-06  
SFT Step    50/5000 | Response PPL:   6.78 | CE: 1.9140 | JEPA: 0.3833 | LR: 1.25e-05  
SFT Step    75/5000 | Response PPL:   6.68 | CE: 1.8998 | JEPA: 0.3660 | LR: 1.88e-05  
SFT Step   100/5000 | Response PPL:   6.51 | CE: 1.8728 | JEPA: 0.3544 | LR: 2.50e-05  
SFT Step   125/5000 | Response PPL:   6.07 | CE: 1.8036 | JEPA: 0.3338 | LR: 3.13e-05  
SFT Step   150/5000 | Response PPL:   6.27 | CE: 1.8359 | JEPA: 0.3389 | LR: 3.75e-05  
SFT Step   175/5000 | Response PPL:   6.21 | CE: 1.8256 | JEPA: 0.3248 | LR: 4.38e-05  
SFT Step   200/5000 | Response PPL:   6.36 | CE: 1.8497 | JEPA: 0.3400 | LR: 5.00e-05  
SFT Step   225/5000 | Response PPL:   5.86 | CE: 1.7677 | JEPA: 0.3405 | LR: 5.00e-05  
SFT Step   250/5000 | Response PPL:   5.81 | CE: 1.7599 | JEPA: 0.3228 | LR: 5.00e-05  
SFT Step   275/5000 | Response PPL:   5.88 | CE: 1.7717 | JEPA: 0.3261 | LR: 5.00e-05  
SFT Step   300/5000 | Response PPL:   5.77 | CE: 1.7525 | JEPA: 0.3205 | LR: 5.00e-05  
SFT Step   325/5000 | Response PPL:   5.69 | CE: 1.7383 | JEPA: 0.3183 | LR: 4.99e-05  
SFT Step   350/5000 | Response PPL:   5.54 | CE: 1.7126 | JEPA: 0.3205 | LR: 4.99e-05  
SFT Step   375/5000 | Response PPL:   5.68 | CE: 1.7367 | JEPA: 0.3236 | LR: 4.99e-05  
SFT Step   400/5000 | Response PPL:   5.66 | CE: 1.7327 | JEPA: 0.3161 | LR: 4.98e-05  
SFT Step   425/5000 | Response PPL:   5.60 | CE: 1.7220 | JEPA: 0.3269 | LR: 4.98e-05  
SFT Step   450/5000 | Response PPL:   5.71 | CE: 1.7417 | JEPA: 0.3125 | LR: 4.97e-05  
SFT Step   475/5000 | Response PPL:   5.45 | CE: 1.6960 | JEPA: 0.3154 | LR: 4.96e-05  
SFT Step   500/5000 | Response PPL:   5.32 | CE: 1.6715 | JEPA: 0.3132 | LR: 4.96e-05  
SFT Step   525/5000 | Response PPL:   5.39 | CE: 1.6850 | JEPA: 0.3163 | LR: 4.95e-05  
SFT Step   550/5000 | Response PPL:   5.11 | CE: 1.6318 | JEPA: 0.3189 | LR: 4.94e-05  
SFT Step   575/5000 | Response PPL:   5.41 | CE: 1.6882 | JEPA: 0.3199 | LR: 4.93e-05  
SFT Step   600/5000 | Response PPL:   5.50 | CE: 1.7047 | JEPA: 0.3199 | LR: 4.92e-05  
SFT Step   625/5000 | Response PPL:   5.24 | CE: 1.6565 | JEPA: 0.3152 | LR: 4.91e-05  
SFT Step   650/5000 | Response PPL:   5.52 | CE: 1.7081 | JEPA: 0.3201 | LR: 4.90e-05  
SFT Step   675/5000 | Response PPL:   5.30 | CE: 1.6668 | JEPA: 0.3109 | LR: 4.89e-05  
SFT Step   700/5000 | Response PPL:   5.31 | CE: 1.6690 | JEPA: 0.3222 | LR: 4.88e-05  
SFT Step   725/5000 | Response PPL:   5.36 | CE: 1.6781 | JEPA: 0.3251 | LR: 4.87e-05  
SFT Step   750/5000 | Response PPL:   5.34 | CE: 1.6754 | JEPA: 0.3143 | LR: 4.86e-05  
SFT Step   775/5000 | Response PPL:   5.37 | CE: 1.6799 | JEPA: 0.3271 | LR: 4.84e-05  
SFT Step   800/5000 | Response PPL:   5.18 | CE: 1.6456 | JEPA: 0.3210 | LR: 4.83e-05  
SFT Step   825/5000 | Response PPL:   5.10 | CE: 1.6290 | JEPA: 0.3111 | LR: 4.81e-05  
SFT Step   850/5000 | Response PPL:   5.21 | CE: 1.6510 | JEPA: 0.3253 | LR: 4.80e-05  
SFT Step   875/5000 | Response PPL:   5.12 | CE: 1.6324 | JEPA: 0.3131 | LR: 4.78e-05  
SFT Step   900/5000 | Response PPL:   5.21 | CE: 1.6497 | JEPA: 0.3188 | LR: 4.77e-05  
SFT Step   925/5000 | Response PPL:   5.08 | CE: 1.6249 | JEPA: 0.3215 | LR: 4.75e-05  
SFT Step   950/5000 | Response PPL:   5.14 | CE: 1.6367 | JEPA: 0.3208 | LR: 4.74e-05  
SFT Step   975/5000 | Response PPL:   5.12 | CE: 1.6333 | JEPA: 0.3220 | LR: 4.72e-05  
SFT Step  1000/5000 | Response PPL:   5.37 | CE: 1.6804 | JEPA: 0.3224 | LR: 4.70e-05  
\--\> Saved SFT Checkpoint: ./checkpoints\_sft\\sft\_step\_1000.pt  
SFT Step  1025/5000 | Response PPL:   5.40 | CE: 1.6857 | JEPA: 0.3246 | LR: 4.68e-05  
SFT Step  1050/5000 | Response PPL:   5.21 | CE: 1.6508 | JEPA: 0.3212 | LR: 4.66e-05  
SFT Step  1075/5000 | Response PPL:   5.06 | CE: 1.6218 | JEPA: 0.3187 | LR: 4.64e-05  
SFT Step  1100/5000 | Response PPL:   5.16 | CE: 1.6413 | JEPA: 0.3280 | LR: 4.62e-05  
SFT Step  1125/5000 | Response PPL:   5.20 | CE: 1.6477 | JEPA: 0.3278 | LR: 4.60e-05  
SFT Step  1150/5000 | Response PPL:   5.39 | CE: 1.6853 | JEPA: 0.3214 | LR: 4.58e-05  
SFT Step  1175/5000 | Response PPL:   5.02 | CE: 1.6131 | JEPA: 0.3160 | LR: 4.56e-05  
SFT Step  1200/5000 | Response PPL:   5.06 | CE: 1.6215 | JEPA: 0.3212 | LR: 4.54e-05  
SFT Step  1225/5000 | Response PPL:   5.26 | CE: 1.6593 | JEPA: 0.3323 | LR: 4.51e-05  
SFT Step  1250/5000 | Response PPL:   5.12 | CE: 1.6331 | JEPA: 0.3195 | LR: 4.49e-05  
SFT Step  1275/5000 | Response PPL:   5.10 | CE: 1.6296 | JEPA: 0.3282 | LR: 4.47e-05  
SFT Step  1300/5000 | Response PPL:   5.21 | CE: 1.6514 | JEPA: 0.3226 | LR: 4.44e-05  
SFT Step  1325/5000 | Response PPL:   4.91 | CE: 1.5917 | JEPA: 0.3249 | LR: 4.42e-05  
SFT Step  1350/5000 | Response PPL:   4.95 | CE: 1.5997 | JEPA: 0.3291 | LR: 4.39e-05  
SFT Step  1375/5000 | Response PPL:   5.13 | CE: 1.6360 | JEPA: 0.3210 | LR: 4.37e-05  
SFT Step  1400/5000 | Response PPL:   5.03 | CE: 1.6156 | JEPA: 0.3201 | LR: 4.34e-05  
SFT Step  1425/5000 | Response PPL:   5.06 | CE: 1.6207 | JEPA: 0.3240 | LR: 4.32e-05  
SFT Step  1450/5000 | Response PPL:   5.22 | CE: 1.6527 | JEPA: 0.3238 | LR: 4.29e-05  
SFT Step  1475/5000 | Response PPL:   5.25 | CE: 1.6590 | JEPA: 0.3209 | LR: 4.26e-05  
SFT Step  1500/5000 | Response PPL:   5.14 | CE: 1.6371 | JEPA: 0.3285 | LR: 4.23e-05  
SFT Step  1525/5000 | Response PPL:   5.03 | CE: 1.6159 | JEPA: 0.3225 | LR: 4.21e-05  
SFT Step  1550/5000 | Response PPL:   5.03 | CE: 1.6161 | JEPA: 0.3226 | LR: 4.18e-05  
SFT Step  1575/5000 | Response PPL:   5.27 | CE: 1.6620 | JEPA: 0.3362 | LR: 4.15e-05  
SFT Step  1600/5000 | Response PPL:   5.06 | CE: 1.6206 | JEPA: 0.3194 | LR: 4.12e-05  
SFT Step  1625/5000 | Response PPL:   5.09 | CE: 1.6277 | JEPA: 0.3238 | LR: 4.09e-05  
SFT Step  1650/5000 | Response PPL:   5.06 | CE: 1.6212 | JEPA: 0.3166 | LR: 4.06e-05  
SFT Step  1675/5000 | Response PPL:   4.87 | CE: 1.5834 | JEPA: 0.3195 | LR: 4.03e-05  
SFT Step  1700/5000 | Response PPL:   4.95 | CE: 1.5993 | JEPA: 0.3106 | LR: 4.00e-05  
SFT Step  1725/5000 | Response PPL:   5.13 | CE: 1.6355 | JEPA: 0.3211 | LR: 3.97e-05  
SFT Step  1750/5000 | Response PPL:   4.99 | CE: 1.6080 | JEPA: 0.3190 | LR: 3.94e-05  
SFT Step  1775/5000 | Response PPL:   5.01 | CE: 1.6112 | JEPA: 0.3205 | LR: 3.91e-05  
SFT Step  1800/5000 | Response PPL:   5.02 | CE: 1.6137 | JEPA: 0.3340 | LR: 3.88e-05  
SFT Step  1825/5000 | Response PPL:   4.84 | CE: 1.5766 | JEPA: 0.3267 | LR: 3.84e-05  
SFT Step  1850/5000 | Response PPL:   4.95 | CE: 1.5987 | JEPA: 0.3296 | LR: 3.81e-05  
SFT Step  1875/5000 | Response PPL:   5.01 | CE: 1.6124 | JEPA: 0.3334 | LR: 3.78e-05  
SFT Step  1900/5000 | Response PPL:   4.95 | CE: 1.5998 | JEPA: 0.3197 | LR: 3.75e-05  
SFT Step  1925/5000 | Response PPL:   5.14 | CE: 1.6370 | JEPA: 0.3265 | LR: 3.71e-05  
SFT Step  1950/5000 | Response PPL:   5.01 | CE: 1.6124 | JEPA: 0.3285 | LR: 3.68e-05  
SFT Step  1975/5000 | Response PPL:   5.09 | CE: 1.6281 | JEPA: 0.3386 | LR: 3.65e-05  
SFT Step  2000/5000 | Response PPL:   4.87 | CE: 1.5830 | JEPA: 0.3187 | LR: 3.61e-05  
\--\> Saved SFT Checkpoint: ./checkpoints\_sft\\sft\_step\_2000.pt  
SFT Step  2025/5000 | Response PPL:   5.11 | CE: 1.6314 | JEPA: 0.3218 | LR: 3.58e-05  
SFT Step  2050/5000 | Response PPL:   4.83 | CE: 1.5757 | JEPA: 0.3317 | LR: 3.54e-05  
SFT Step  2075/5000 | Response PPL:   5.04 | CE: 1.6178 | JEPA: 0.3242 | LR: 3.51e-05  
SFT Step  2100/5000 | Response PPL:   4.90 | CE: 1.5883 | JEPA: 0.3243 | LR: 3.47e-05  
SFT Step  2125/5000 | Response PPL:   4.96 | CE: 1.6021 | JEPA: 0.3220 | LR: 3.44e-05  
SFT Step  2150/5000 | Response PPL:   5.00 | CE: 1.6094 | JEPA: 0.3309 | LR: 3.40e-05  
SFT Step  2175/5000 | Response PPL:   4.99 | CE: 1.6080 | JEPA: 0.3360 | LR: 3.37e-05  
SFT Step  2200/5000 | Response PPL:   4.74 | CE: 1.5569 | JEPA: 0.3145 | LR: 3.33e-05  
SFT Step  2225/5000 | Response PPL:   4.99 | CE: 1.6065 | JEPA: 0.3317 | LR: 3.30e-05  
SFT Step  2250/5000 | Response PPL:   4.93 | CE: 1.5963 | JEPA: 0.3145 | LR: 3.26e-05  
SFT Step  2275/5000 | Response PPL:   4.97 | CE: 1.6037 | JEPA: 0.3255 | LR: 3.23e-05  
SFT Step  2300/5000 | Response PPL:   4.93 | CE: 1.5963 | JEPA: 0.3264 | LR: 3.19e-05  
SFT Step  2325/5000 | Response PPL:   4.93 | CE: 1.5960 | JEPA: 0.3275 | LR: 3.15e-05  
SFT Step  2350/5000 | Response PPL:   5.02 | CE: 1.6131 | JEPA: 0.3283 | LR: 3.12e-05  
SFT Step  2375/5000 | Response PPL:   5.07 | CE: 1.6231 | JEPA: 0.3260 | LR: 3.08e-05  
SFT Step  2400/5000 | Response PPL:   4.95 | CE: 1.6002 | JEPA: 0.3267 | LR: 3.05e-05  
SFT Step  2425/5000 | Response PPL:   4.91 | CE: 1.5918 | JEPA: 0.3233 | LR: 3.01e-05  
SFT Step  2450/5000 | Response PPL:   5.02 | CE: 1.6136 | JEPA: 0.3248 | LR: 2.97e-05  
SFT Step  2475/5000 | Response PPL:   4.81 | CE: 1.5707 | JEPA: 0.3245 | LR: 2.94e-05  
SFT Step  2500/5000 | Response PPL:   5.22 | CE: 1.6516 | JEPA: 0.3330 | LR: 2.90e-05  
SFT Step  2525/5000 | Response PPL:   4.81 | CE: 1.5701 | JEPA: 0.3277 | LR: 2.86e-05  
SFT Step  2550/5000 | Response PPL:   4.90 | CE: 1.5886 | JEPA: 0.3275 | LR: 2.83e-05  
SFT Step  2575/5000 | Response PPL:   4.84 | CE: 1.5774 | JEPA: 0.3362 | LR: 2.79e-05  
SFT Step  2600/5000 | Response PPL:   4.89 | CE: 1.5871 | JEPA: 0.3208 | LR: 2.75e-05  
SFT Step  2625/5000 | Response PPL:   4.95 | CE: 1.6001 | JEPA: 0.3321 | LR: 2.71e-05  
SFT Step  2650/5000 | Response PPL:   4.92 | CE: 1.5942 | JEPA: 0.3186 | LR: 2.68e-05  
SFT Step  2675/5000 | Response PPL:   4.71 | CE: 1.5486 | JEPA: 0.3201 | LR: 2.64e-05  
SFT Step  2700/5000 | Response PPL:   4.80 | CE: 1.5693 | JEPA: 0.3153 | LR: 2.60e-05  
SFT Step  2725/5000 | Response PPL:   4.93 | CE: 1.5961 | JEPA: 0.3358 | LR: 2.57e-05  
SFT Step  2750/5000 | Response PPL:   4.99 | CE: 1.6070 | JEPA: 0.3362 | LR: 2.53e-05  
SFT Step  2775/5000 | Response PPL:   4.74 | CE: 1.5560 | JEPA: 0.3215 | LR: 2.49e-05  
SFT Step  2800/5000 | Response PPL:   4.97 | CE: 1.6043 | JEPA: 0.3298 | LR: 2.46e-05  
SFT Step  2825/5000 | Response PPL:   4.77 | CE: 1.5615 | JEPA: 0.3361 | LR: 2.42e-05  
SFT Step  2850/5000 | Response PPL:   4.75 | CE: 1.5583 | JEPA: 0.3160 | LR: 2.38e-05  
SFT Step  2875/5000 | Response PPL:   4.86 | CE: 1.5818 | JEPA: 0.3180 | LR: 2.35e-05  
SFT Step  2900/5000 | Response PPL:   4.91 | CE: 1.5921 | JEPA: 0.3249 | LR: 2.31e-05  
SFT Step  2925/5000 | Response PPL:   4.78 | CE: 1.5651 | JEPA: 0.3243 | LR: 2.28e-05  
SFT Step  2950/5000 | Response PPL:   4.98 | CE: 1.6051 | JEPA: 0.3253 | LR: 2.24e-05  
SFT Step  2975/5000 | Response PPL:   4.79 | CE: 1.5660 | JEPA: 0.3204 | LR: 2.20e-05  
SFT Step  3000/5000 | Response PPL:   4.97 | CE: 1.6026 | JEPA: 0.3250 | LR: 2.17e-05  
\--\> Saved SFT Checkpoint: ./checkpoints\_sft\\sft\_step\_3000.pt  
SFT Step  3025/5000 | Response PPL:   4.83 | CE: 1.5757 | JEPA: 0.3239 | LR: 2.13e-05  
SFT Step  3050/5000 | Response PPL:   4.94 | CE: 1.5970 | JEPA: 0.3292 | LR: 2.10e-05  
SFT Step  3075/5000 | Response PPL:   4.83 | CE: 1.5743 | JEPA: 0.3263 | LR: 2.06e-05  
SFT Step  3100/5000 | Response PPL:   5.04 | CE: 1.6180 | JEPA: 0.3360 | LR: 2.03e-05  
SFT Step  3125/5000 | Response PPL:   4.90 | CE: 1.5884 | JEPA: 0.3364 | LR: 1.99e-05  
SFT Step  3150/5000 | Response PPL:   4.86 | CE: 1.5819 | JEPA: 0.3249 | LR: 1.96e-05  
SFT Step  3175/5000 | Response PPL:   4.87 | CE: 1.5821 | JEPA: 0.3279 | LR: 1.92e-05  
SFT Step  3200/5000 | Response PPL:   4.66 | CE: 1.5395 | JEPA: 0.3187 | LR: 1.89e-05  
SFT Step  3225/5000 | Response PPL:   4.76 | CE: 1.5592 | JEPA: 0.3292 | LR: 1.86e-05  
SFT Step  3250/5000 | Response PPL:   5.01 | CE: 1.6112 | JEPA: 0.3324 | LR: 1.82e-05  
SFT Step  3275/5000 | Response PPL:   5.11 | CE: 1.6309 | JEPA: 0.3360 | LR: 1.79e-05  
SFT Step  3300/5000 | Response PPL:   4.72 | CE: 1.5514 | JEPA: 0.3246 | LR: 1.76e-05  
SFT Step  3325/5000 | Response PPL:   4.77 | CE: 1.5633 | JEPA: 0.3302 | LR: 1.72e-05  
SFT Step  3350/5000 | Response PPL:   4.63 | CE: 1.5328 | JEPA: 0.3245 | LR: 1.69e-05  
SFT Step  3375/5000 | Response PPL:   4.72 | CE: 1.5528 | JEPA: 0.3290 | LR: 1.66e-05  
SFT Step  3400/5000 | Response PPL:   4.74 | CE: 1.5555 | JEPA: 0.3253 | LR: 1.63e-05  
SFT Step  3425/5000 | Response PPL:   4.66 | CE: 1.5389 | JEPA: 0.3149 | LR: 1.59e-05  
SFT Step  3450/5000 | Response PPL:   4.76 | CE: 1.5612 | JEPA: 0.3103 | LR: 1.56e-05  
SFT Step  3475/5000 | Response PPL:   4.98 | CE: 1.6062 | JEPA: 0.3283 | LR: 1.53e-05  
SFT Step  3500/5000 | Response PPL:   4.75 | CE: 1.5589 | JEPA: 0.3256 | LR: 1.50e-05  
SFT Step  3525/5000 | Response PPL:   4.77 | CE: 1.5626 | JEPA: 0.3328 | LR: 1.47e-05  
SFT Step  3550/5000 | Response PPL:   4.79 | CE: 1.5671 | JEPA: 0.3248 | LR: 1.44e-05  
SFT Step  3575/5000 | Response PPL:   4.84 | CE: 1.5760 | JEPA: 0.3227 | LR: 1.41e-05  
SFT Step  3600/5000 | Response PPL:   4.83 | CE: 1.5743 | JEPA: 0.3206 | LR: 1.38e-05  
SFT Step  3625/5000 | Response PPL:   4.72 | CE: 1.5512 | JEPA: 0.3278 | LR: 1.35e-05  
SFT Step  3650/5000 | Response PPL:   4.96 | CE: 1.6014 | JEPA: 0.3290 | LR: 1.32e-05  
SFT Step  3675/5000 | Response PPL:   4.79 | CE: 1.5660 | JEPA: 0.3148 | LR: 1.30e-05  
SFT Step  3700/5000 | Response PPL:   5.07 | CE: 1.6227 | JEPA: 0.3258 | LR: 1.27e-05  
SFT Step  3725/5000 | Response PPL:   4.68 | CE: 1.5439 | JEPA: 0.3266 | LR: 1.24e-05  
SFT Step  3750/5000 | Response PPL:   4.97 | CE: 1.6043 | JEPA: 0.3347 | LR: 1.21e-05  
SFT Step  3775/5000 | Response PPL:   4.98 | CE: 1.6050 | JEPA: 0.3290 | LR: 1.19e-05  
SFT Step  3800/5000 | Response PPL:   4.58 | CE: 1.5215 | JEPA: 0.3155 | LR: 1.16e-05  
SFT Step  3825/5000 | Response PPL:   4.58 | CE: 1.5208 | JEPA: 0.3206 | LR: 1.13e-05  
SFT Step  3850/5000 | Response PPL:   4.89 | CE: 1.5867 | JEPA: 0.3351 | LR: 1.11e-05  
SFT Step  3875/5000 | Response PPL:   4.62 | CE: 1.5305 | JEPA: 0.3267 | LR: 1.08e-05  
SFT Step  3900/5000 | Response PPL:   4.79 | CE: 1.5668 | JEPA: 0.3335 | LR: 1.06e-05  
SFT Step  3925/5000 | Response PPL:   4.79 | CE: 1.5657 | JEPA: 0.3264 | LR: 1.04e-05  
SFT Step  3950/5000 | Response PPL:   4.74 | CE: 1.5571 | JEPA: 0.3416 | LR: 1.01e-05  
SFT Step  3975/5000 | Response PPL:   4.78 | CE: 1.5650 | JEPA: 0.3234 | LR: 9.89e-06  
SFT Step  4000/5000 | Response PPL:   4.80 | CE: 1.5687 | JEPA: 0.3396 | LR: 9.66e-06  
\--\> Saved SFT Checkpoint: ./checkpoints\_sft\\sft\_step\_4000.pt  
SFT Step  4025/5000 | Response PPL:   4.89 | CE: 1.5878 | JEPA: 0.3355 | LR: 9.44e-06  
SFT Step  4050/5000 | Response PPL:   4.72 | CE: 1.5515 | JEPA: 0.3179 | LR: 9.22e-06  
SFT Step  4075/5000 | Response PPL:   4.79 | CE: 1.5673 | JEPA: 0.3291 | LR: 9.01e-06  
SFT Step  4100/5000 | Response PPL:   4.78 | CE: 1.5652 | JEPA: 0.3236 | LR: 8.80e-06  
SFT Step  4125/5000 | Response PPL:   4.68 | CE: 1.5441 | JEPA: 0.3304 | LR: 8.60e-06  
SFT Step  4150/5000 | Response PPL:   4.78 | CE: 1.5643 | JEPA: 0.3248 | LR: 8.40e-06  
SFT Step  4175/5000 | Response PPL:   4.77 | CE: 1.5628 | JEPA: 0.3310 | LR: 8.21e-06  
SFT Step  4200/5000 | Response PPL:   4.64 | CE: 1.5354 | JEPA: 0.3261 | LR: 8.02e-06  
SFT Step  4225/5000 | Response PPL:   4.74 | CE: 1.5562 | JEPA: 0.3310 | LR: 7.84e-06  
SFT Step  4250/5000 | Response PPL:   4.79 | CE: 1.5655 | JEPA: 0.3182 | LR: 7.66e-06  
SFT Step  4275/5000 | Response PPL:   4.84 | CE: 1.5778 | JEPA: 0.3251 | LR: 7.49e-06  
SFT Step  4300/5000 | Response PPL:   4.70 | CE: 1.5480 | JEPA: 0.3226 | LR: 7.33e-06  
SFT Step  4325/5000 | Response PPL:   4.73 | CE: 1.5537 | JEPA: 0.3187 | LR: 7.17e-06  
SFT Step  4350/5000 | Response PPL:   4.73 | CE: 1.5548 | JEPA: 0.3216 | LR: 7.01e-06  
SFT Step  4375/5000 | Response PPL:   4.69 | CE: 1.5455 | JEPA: 0.3243 | LR: 6.86e-06  
SFT Step  4400/5000 | Response PPL:   4.86 | CE: 1.5808 | JEPA: 0.3246 | LR: 6.72e-06  
SFT Step  4425/5000 | Response PPL:   4.75 | CE: 1.5571 | JEPA: 0.3083 | LR: 6.58e-06  
SFT Step  4450/5000 | Response PPL:   4.64 | CE: 1.5337 | JEPA: 0.3201 | LR: 6.45e-06  
SFT Step  4475/5000 | Response PPL:   4.62 | CE: 1.5303 | JEPA: 0.3215 | LR: 6.32e-06  
SFT Step  4500/5000 | Response PPL:   4.82 | CE: 1.5736 | JEPA: 0.3247 | LR: 6.20e-06  
SFT Step  4525/5000 | Response PPL:   4.70 | CE: 1.5467 | JEPA: 0.3218 | LR: 6.08e-06  
SFT Step  4550/5000 | Response PPL:   4.72 | CE: 1.5515 | JEPA: 0.3364 | LR: 5.97e-06  
SFT Step  4575/5000 | Response PPL:   4.70 | CE: 1.5466 | JEPA: 0.3283 | LR: 5.87e-06  
SFT Step  4600/5000 | Response PPL:   4.68 | CE: 1.5440 | JEPA: 0.3207 | LR: 5.77e-06  
SFT Step  4625/5000 | Response PPL:   4.64 | CE: 1.5344 | JEPA: 0.3187 | LR: 5.68e-06  
SFT Step  4650/5000 | Response PPL:   4.74 | CE: 1.5569 | JEPA: 0.3209 | LR: 5.59e-06  
SFT Step  4675/5000 | Response PPL:   4.60 | CE: 1.5270 | JEPA: 0.3166 | LR: 5.51e-06  
SFT Step  4700/5000 | Response PPL:   4.80 | CE: 1.5688 | JEPA: 0.3214 | LR: 5.44e-06  
SFT Step  4725/5000 | Response PPL:   4.82 | CE: 1.5726 | JEPA: 0.3315 | LR: 5.37e-06  
SFT Step  4750/5000 | Response PPL:   4.65 | CE: 1.5379 | JEPA: 0.3239 | LR: 5.30e-06  
SFT Step  4775/5000 | Response PPL:   4.59 | CE: 1.5236 | JEPA: 0.3229 | LR: 5.25e-06  
SFT Step  4800/5000 | Response PPL:   4.80 | CE: 1.5677 | JEPA: 0.3245 | LR: 5.19e-06  
SFT Step  4825/5000 | Response PPL:   4.70 | CE: 1.5484 | JEPA: 0.3228 | LR: 5.15e-06  
SFT Step  4850/5000 | Response PPL:   4.65 | CE: 1.5371 | JEPA: 0.3179 | LR: 5.11e-06  
SFT Step  4875/5000 | Response PPL:   4.81 | CE: 1.5707 | JEPA: 0.3330 | LR: 5.08e-06  
SFT Step  4900/5000 | Response PPL:   4.95 | CE: 1.5988 | JEPA: 0.3285 | LR: 5.05e-06  
SFT Step  4925/5000 | Response PPL:   4.62 | CE: 1.5311 | JEPA: 0.3310 | LR: 5.03e-06  
SFT Step  4950/5000 | Response PPL:   4.61 | CE: 1.5284 | JEPA: 0.3253 | LR: 5.01e-06  
SFT Step  4975/5000 | Response PPL:   4.83 | CE: 1.5747 | JEPA: 0.3341 | LR: 5.00e-06  
SFT Step  5000/5000 | Response PPL:   4.85 | CE: 1.5795 | JEPA: 0.3320 | LR: 5.00e-06  
\--\> Saved SFT Checkpoint: ./checkpoints\_sft\\sft\_step\_5000.pt

**COT tuning after SFT with x .1 bayesian centroid lr** 

\[transformers\] Token indices sequence length is longer than the specified maximum sequence length for this model (1223 \> 1024). Running this sequence through the model will result in indexing errors  
CoT Step   25/3000 | Response PPL:   6.76 | CE: 1.9109 | JEPA: 0.1493 | LR: 6.67e-06  
CoT Step   50/3000 | Response PPL:   4.71 | CE: 1.5506 | JEPA: 0.1392 | LR: 1.33e-05  
CoT Step   75/3000 | Response PPL:   4.45 | CE: 1.4937 | JEPA: 0.1483 | LR: 2.00e-05  
CoT Step  100/3000 | Response PPL:   4.30 | CE: 1.4589 | JEPA: 0.1425 | LR: 2.67e-05  
CoT Step  125/3000 | Response PPL:   4.31 | CE: 1.4613 | JEPA: 0.1416 | LR: 3.33e-05  
CoT Step  150/3000 | Response PPL:   4.35 | CE: 1.4693 | JEPA: 0.1415 | LR: 4.00e-05  
CoT Step  175/3000 | Response PPL:   4.45 | CE: 1.4929 | JEPA: 0.1398 | LR: 4.00e-05  
CoT Step  200/3000 | Response PPL:   4.12 | CE: 1.4157 | JEPA: 0.1351 | LR: 4.00e-05  
CoT Step  225/3000 | Response PPL:   3.92 | CE: 1.3673 | JEPA: 0.1384 | LR: 3.99e-05  
CoT Step  250/3000 | Response PPL:   4.01 | CE: 1.3888 | JEPA: 0.1404 | LR: 3.99e-05  
CoT Step  275/3000 | Response PPL:   3.98 | CE: 1.3804 | JEPA: 0.1384 | LR: 3.98e-05  
CoT Step  300/3000 | Response PPL:   4.04 | CE: 1.3952 | JEPA: 0.1379 | LR: 3.98e-05  
CoT Step  325/3000 | Response PPL:   3.95 | CE: 1.3738 | JEPA: 0.1366 | LR: 3.97e-05  
CoT Step  350/3000 | Response PPL:   4.01 | CE: 1.3883 | JEPA: 0.1366 | LR: 3.96e-05  
CoT Step  375/3000 | Response PPL:   3.99 | CE: 1.3833 | JEPA: 0.1366 | LR: 3.95e-05  
CoT Step  400/3000 | Response PPL:   4.07 | CE: 1.4036 | JEPA: 0.1375 | LR: 3.93e-05  
CoT Step  425/3000 | Response PPL:   3.84 | CE: 1.3459 | JEPA: 0.1361 | LR: 3.92e-05  
CoT Step  450/3000 | Response PPL:   4.14 | CE: 1.4212 | JEPA: 0.1446 | LR: 3.90e-05  
CoT Step  475/3000 | Response PPL:   3.63 | CE: 1.2880 | JEPA: 0.1341 | LR: 3.89e-05  
CoT Step  500/3000 | Response PPL:   3.68 | CE: 1.3027 | JEPA: 0.1346 | LR: 3.87e-05  
CoT Step  525/3000 | Response PPL:   3.54 | CE: 1.2650 | JEPA: 0.1363 | LR: 3.85e-05  
CoT Step  550/3000 | Response PPL:   3.61 | CE: 1.2838 | JEPA: 0.1401 | LR: 3.83e-05  
CoT Step  575/3000 | Response PPL:   3.71 | CE: 1.3116 | JEPA: 0.1382 | LR: 3.81e-05  
CoT Step  600/3000 | Response PPL:   3.41 | CE: 1.2256 | JEPA: 0.1368 | LR: 3.78e-05  
CoT Step  625/3000 | Response PPL:   3.54 | CE: 1.2647 | JEPA: 0.1391 | LR: 3.76e-05  
CoT Step  650/3000 | Response PPL:   3.69 | CE: 1.3062 | JEPA: 0.1388 | LR: 3.73e-05  
CoT Step  675/3000 | Response PPL:   3.72 | CE: 1.3129 | JEPA: 0.1376 | LR: 3.71e-05  
CoT Step  700/3000 | Response PPL:   3.60 | CE: 1.2808 | JEPA: 0.1355 | LR: 3.68e-05  
CoT Step  725/3000 | Response PPL:   3.67 | CE: 1.2991 | JEPA: 0.1374 | LR: 3.65e-05  
CoT Step  750/3000 | Response PPL:   3.46 | CE: 1.2410 | JEPA: 0.1367 | LR: 3.62e-05  
CoT Step  775/3000 | Response PPL:   3.49 | CE: 1.2504 | JEPA: 0.1350 | LR: 3.59e-05  
CoT Step  800/3000 | Response PPL:   3.44 | CE: 1.2354 | JEPA: 0.1314 | LR: 3.56e-05  
CoT Step  825/3000 | Response PPL:   3.50 | CE: 1.2533 | JEPA: 0.1387 | LR: 3.53e-05  
CoT Step  850/3000 | Response PPL:   3.39 | CE: 1.2204 | JEPA: 0.1350 | LR: 3.49e-05  
CoT Step  875/3000 | Response PPL:   3.29 | CE: 1.1900 | JEPA: 0.1370 | LR: 3.46e-05  
CoT Step  900/3000 | Response PPL:   3.54 | CE: 1.2636 | JEPA: 0.1355 | LR: 3.42e-05  
CoT Step  925/3000 | Response PPL:   3.42 | CE: 1.2291 | JEPA: 0.1327 | LR: 3.38e-05  
CoT Step  950/3000 | Response PPL:   3.75 | CE: 1.3213 | JEPA: 0.1459 | LR: 3.35e-05  
CoT Step  975/3000 | Response PPL:   3.27 | CE: 1.1834 | JEPA: 0.1389 | LR: 3.31e-05  
CoT Step 1000/3000 | Response PPL:   3.25 | CE: 1.1779 | JEPA: 0.1391 | LR: 3.27e-05  
\--\> Saved CoT Checkpoint: ./checkpoints\_cot\_sft\\cot\_step\_1000.pt  
CoT Step 1025/3000 | Response PPL:   3.35 | CE: 1.2100 | JEPA: 0.1393 | LR: 3.23e-05  
CoT Step 1050/3000 | Response PPL:   3.31 | CE: 1.1974 | JEPA: 0.1446 | LR: 3.19e-05  
CoT Step 1075/3000 | Response PPL:   3.39 | CE: 1.2222 | JEPA: 0.1357 | LR: 3.14e-05  
CoT Step 1100/3000 | Response PPL:   3.42 | CE: 1.2311 | JEPA: 0.1397 | LR: 3.10e-05  
CoT Step 1125/3000 | Response PPL:   3.49 | CE: 1.2512 | JEPA: 0.1425 | LR: 3.06e-05  
CoT Step 1150/3000 | Response PPL:   3.38 | CE: 1.2187 | JEPA: 0.1391 | LR: 3.01e-05  
CoT Step 1175/3000 | Response PPL:   3.43 | CE: 1.2326 | JEPA: 0.1377 | LR: 2.97e-05  
CoT Step 1200/3000 | Response PPL:   3.24 | CE: 1.1749 | JEPA: 0.1365 | LR: 2.92e-05  
CoT Step 1225/3000 | Response PPL:   3.22 | CE: 1.1708 | JEPA: 0.1402 | LR: 2.88e-05  
CoT Step 1250/3000 | Response PPL:   3.25 | CE: 1.1794 | JEPA: 0.1405 | LR: 2.83e-05  
CoT Step 1275/3000 | Response PPL:   3.32 | CE: 1.2008 | JEPA: 0.1424 | LR: 2.79e-05  
CoT Step 1300/3000 | Response PPL:   3.28 | CE: 1.1870 | JEPA: 0.1407 | LR: 2.74e-05  
CoT Step 1325/3000 | Response PPL:   3.21 | CE: 1.1652 | JEPA: 0.1473 | LR: 2.69e-05  
CoT Step 1350/3000 | Response PPL:   3.19 | CE: 1.1613 | JEPA: 0.1425 | LR: 2.64e-05  
CoT Step 1375/3000 | Response PPL:   3.23 | CE: 1.1717 | JEPA: 0.1417 | LR: 2.60e-05  
CoT Step 1400/3000 | Response PPL:   3.32 | CE: 1.2007 | JEPA: 0.1447 | LR: 2.55e-05  
CoT Step 1425/3000 | Response PPL:   3.09 | CE: 1.1272 | JEPA: 0.1429 | LR: 2.50e-05  
CoT Step 1450/3000 | Response PPL:   3.22 | CE: 1.1703 | JEPA: 0.1447 | LR: 2.45e-05  
CoT Step 1475/3000 | Response PPL:   3.23 | CE: 1.1714 | JEPA: 0.1440 | LR: 2.40e-05  
CoT Step 1500/3000 | Response PPL:   3.28 | CE: 1.1865 | JEPA: 0.1448 | LR: 2.35e-05  
CoT Step 1525/3000 | Response PPL:   3.27 | CE: 1.1840 | JEPA: 0.1394 | LR: 2.30e-05  
CoT Step 1550/3000 | Response PPL:   3.17 | CE: 1.1536 | JEPA: 0.1416 | LR: 2.25e-05  
CoT Step 1575/3000 | Response PPL:   3.28 | CE: 1.1864 | JEPA: 0.1414 | LR: 2.20e-05  
CoT Step 1600/3000 | Response PPL:   3.39 | CE: 1.2210 | JEPA: 0.1474 | LR: 2.15e-05  
CoT Step 1625/3000 | Response PPL:   3.14 | CE: 1.1450 | JEPA: 0.1433 | LR: 2.10e-05  
CoT Step 1650/3000 | Response PPL:   3.09 | CE: 1.1271 | JEPA: 0.1392 | LR: 2.05e-05  
CoT Step 1675/3000 | Response PPL:   3.34 | CE: 1.2065 | JEPA: 0.1477 | LR: 2.00e-05  
CoT Step 1700/3000 | Response PPL:   3.13 | CE: 1.1395 | JEPA: 0.1449 | LR: 1.95e-05  
CoT Step 1725/3000 | Response PPL:   3.23 | CE: 1.1717 | JEPA: 0.1454 | LR: 1.91e-05  
CoT Step 1750/3000 | Response PPL:   3.07 | CE: 1.1209 | JEPA: 0.1416 | LR: 1.86e-05  
CoT Step 1775/3000 | Response PPL:   3.27 | CE: 1.1852 | JEPA: 0.1448 | LR: 1.81e-05  
CoT Step 1800/3000 | Response PPL:   3.13 | CE: 1.1411 | JEPA: 0.1419 | LR: 1.76e-05  
CoT Step 1825/3000 | Response PPL:   3.20 | CE: 1.1622 | JEPA: 0.1501 | LR: 1.71e-05  
CoT Step 1850/3000 | Response PPL:   3.06 | CE: 1.1178 | JEPA: 0.1412 | LR: 1.66e-05  
CoT Step 1875/3000 | Response PPL:   3.18 | CE: 1.1576 | JEPA: 0.1462 | LR: 1.62e-05  
CoT Step 1900/3000 | Response PPL:   3.16 | CE: 1.1499 | JEPA: 0.1460 | LR: 1.57e-05  
CoT Step 1925/3000 | Response PPL:   3.13 | CE: 1.1415 | JEPA: 0.1449 | LR: 1.52e-05  
CoT Step 1950/3000 | Response PPL:   3.41 | CE: 1.2253 | JEPA: 0.1491 | LR: 1.48e-05  
CoT Step 1975/3000 | Response PPL:   2.92 | CE: 1.0701 | JEPA: 0.1402 | LR: 1.43e-05  
CoT Step 2000/3000 | Response PPL:   2.97 | CE: 1.0900 | JEPA: 0.1440 | LR: 1.39e-05  
\--\> Saved CoT Checkpoint: ./checkpoints\_cot\_sft\\cot\_step\_2000.pt  
CoT Step 2025/3000 | Response PPL:   3.31 | CE: 1.1963 | JEPA: 0.1506 | LR: 1.35e-05  
CoT Step 2050/3000 | Response PPL:   3.11 | CE: 1.1342 | JEPA: 0.1458 | LR: 1.30e-05  
CoT Step 2075/3000 | Response PPL:   3.21 | CE: 1.1673 | JEPA: 0.1486 | LR: 1.26e-05  
CoT Step 2100/3000 | Response PPL:   2.91 | CE: 1.0689 | JEPA: 0.1434 | LR: 1.22e-05  
CoT Step 2125/3000 | Response PPL:   3.11 | CE: 1.1347 | JEPA: 0.1486 | LR: 1.18e-05  
CoT Step 2150/3000 | Response PPL:   3.05 | CE: 1.1158 | JEPA: 0.1504 | LR: 1.14e-05  
CoT Step 2175/3000 | Response PPL:   2.91 | CE: 1.0676 | JEPA: 0.1447 | LR: 1.10e-05  
CoT Step 2200/3000 | Response PPL:   3.23 | CE: 1.1727 | JEPA: 0.1492 | LR: 1.06e-05  
CoT Step 2225/3000 | Response PPL:   3.13 | CE: 1.1398 | JEPA: 0.1483 | LR: 1.02e-05  
CoT Step 2250/3000 | Response PPL:   3.14 | CE: 1.1440 | JEPA: 0.1479 | LR: 9.82e-06  
CoT Step 2275/3000 | Response PPL:   2.99 | CE: 1.0940 | JEPA: 0.1423 | LR: 9.46e-06  
CoT Step 2300/3000 | Response PPL:   2.91 | CE: 1.0681 | JEPA: 0.1431 | LR: 9.11e-06  
CoT Step 2325/3000 | Response PPL:   2.91 | CE: 1.0681 | JEPA: 0.1471 | LR: 8.77e-06  
CoT Step 2350/3000 | Response PPL:   3.16 | CE: 1.1518 | JEPA: 0.1449 | LR: 8.44e-06  
CoT Step 2375/3000 | Response PPL:   3.05 | CE: 1.1168 | JEPA: 0.1468 | LR: 8.12e-06  
CoT Step 2400/3000 | Response PPL:   2.99 | CE: 1.0947 | JEPA: 0.1445 | LR: 7.81e-06  
CoT Step 2425/3000 | Response PPL:   3.17 | CE: 1.1545 | JEPA: 0.1482 | LR: 7.51e-06  
CoT Step 2450/3000 | Response PPL:   2.98 | CE: 1.0906 | JEPA: 0.1397 | LR: 7.22e-06  
CoT Step 2475/3000 | Response PPL:   2.93 | CE: 1.0746 | JEPA: 0.1476 | LR: 6.94e-06  
CoT Step 2500/3000 | Response PPL:   2.93 | CE: 1.0750 | JEPA: 0.1474 | LR: 6.68e-06  
CoT Step 2525/3000 | Response PPL:   2.78 | CE: 1.0238 | JEPA: 0.1472 | LR: 6.42e-06  
CoT Step 2550/3000 | Response PPL:   3.00 | CE: 1.0999 | JEPA: 0.1500 | LR: 6.18e-06  
CoT Step 2575/3000 | Response PPL:   3.17 | CE: 1.1549 | JEPA: 0.1491 | LR: 5.95e-06  
CoT Step 2600/3000 | Response PPL:   2.86 | CE: 1.0502 | JEPA: 0.1443 | LR: 5.73e-06  
CoT Step 2625/3000 | Response PPL:   3.02 | CE: 1.1054 | JEPA: 0.1517 | LR: 5.52e-06  
CoT Step 2650/3000 | Response PPL:   3.01 | CE: 1.1020 | JEPA: 0.1462 | LR: 5.33e-06  
CoT Step 2675/3000 | Response PPL:   2.92 | CE: 1.0702 | JEPA: 0.1458 | LR: 5.15e-06  
CoT Step 2700/3000 | Response PPL:   2.90 | CE: 1.0631 | JEPA: 0.1500 | LR: 4.98e-06  
CoT Step 2725/3000 | Response PPL:   2.96 | CE: 1.0843 | JEPA: 0.1462 | LR: 4.83e-06  
CoT Step 2750/3000 | Response PPL:   3.03 | CE: 1.1102 | JEPA: 0.1492 | LR: 4.68e-06  
CoT Step 2775/3000 | Response PPL:   3.00 | CE: 1.1002 | JEPA: 0.1473 | LR: 4.56e-06  
CoT Step 2800/3000 | Response PPL:   3.37 | CE: 1.2148 | JEPA: 0.1535 | LR: 4.44e-06  
CoT Step 2825/3000 | Response PPL:   2.99 | CE: 1.0944 | JEPA: 0.1407 | LR: 4.34e-06  
CoT Step 2850/3000 | Response PPL:   3.03 | CE: 1.1084 | JEPA: 0.1477 | LR: 4.25e-06  
CoT Step 2875/3000 | Response PPL:   3.19 | CE: 1.1598 | JEPA: 0.1511 | LR: 4.17e-06  
CoT Step 2900/3000 | Response PPL:   2.79 | CE: 1.0252 | JEPA: 0.1498 | LR: 4.11e-06  
CoT Step 2925/3000 | Response PPL:   2.96 | CE: 1.0839 | JEPA: 0.1464 | LR: 4.06e-06  
CoT Step 2950/3000 | Response PPL:   2.88 | CE: 1.0574 | JEPA: 0.1470 | LR: 4.03e-06  
CoT Step 2975/3000 | Response PPL:   2.96 | CE: 1.0840 | JEPA: 0.1577 | LR: 4.01e-06  
CoT Step 3000/3000 | Response PPL:   3.24 | CE: 1.1769 | JEPA: 0.1493 | LR: 4.00e-06  
\--\> Saved CoT Checkpoint: ./checkpoints\_cot\_sft\\cot\_step\_3000.pt

**Cosine similarity on original datasets seem inaccurate relative to qualitative output** 

✅ Successfully loaded model from Step 3000\!

\=================================================================  
🧠 BAYESIAN CENTROID MEMORY DIAGNOSTICS  
\=================================================================  
Total Prototypes:         512  
Average Variance (σ²):    1.0090  (Prior \= 1.0000)  
Min Variance (Sharpest):  0.7293  
Max Variance (Diffuse):   1.3032  
Prototype Norm Mean:      1.3413  
Active Memory Spread:     0.4383

\=================================================================  
🗺️ COGNITIVE MAP & JEPA MECHANISM PROBE  
\=================================================================  
Cognitive Map Step Velocity (Avg): 2.5006  
Total Manifold Path Displacement:   29.1133  
Centroids Activated by Prompt:      7/512 unique clusters  
JEPA Next-Latent Cosine Sim:        0.5609  (56.1% alignment)

\=================================================================  
⚡ GENERATIVE PROBES & HARDWARE THROUGHPUT BENCHMARKS  
Sampling: Temp=0.65 | Top-P=0.85 | Top-K=40 | Rep-Penalty=1.2  
\=================================================================

\--- Domain: Web/Educational \---  
The theory of general relativity explains that gravity is not the same thing as it seems. The reason for this is that our understanding of gravity is an abstract concept, and we are trying to understand why things are the way they are in the first place.

One of the most fascinating aspects of quantum mechanics is its connection to the concept of time dilation. In quantum mechanics, time dilation refers to the phenomenon where particles can be reduced to their individual states at different scales, rather than being reduced to a single state. This idea has been explored extensively by scientists who have been studying the behavior of particles with a large scale, such as those found in nature.  
\</think\>  
The study of time dilation has also led to important advances in our understanding of the universe's structure and evolution. The concept of time dilation has significant implications for our understanding of the origins and behavior of the universe, including the role of gravity in shaping the fabric of spacetime.\<|im\_end|\>

One of the most intriguing aspects of quantum mechanics is its connection to the concept of gauge theory, which studies the behavior of particles at different scales. In quantum mechanics, gauge theory describes how particles can be reduced to their individual states based on the size and distribution of their wave functions. This idea has been explored by scientists who have been studying the behavior of particles with a large scale, such as those found in nature. These researchers have also made significant progress in understanding the behavior of particles with a large scale, such as those found in  
\-----------------------------------------------------------------  
  📥 Input Speed (Prefill):    1,812.9 tok/s  (9 tokens in 4.96 ms)  
  📤 Output Speed (Decode):      182.7 tok/s  (300 tokens in 1.64 s)  
  ⏱️  Latency per Token:          5.47 ms/token  
\-----------------------------------------------------------------

\--- Domain: Synthetic Reasoning \---  
Photosynthesis is crucial for life on Earth because it provides energy, oxygen and carbon dioxide. The process of photosynthesis is a key part of the natural cycle of life.

Soil is another vital resource that plays an important role in maintaining the health of ecosystems. Soil is a rich source of essential nutrients such as nitrogen, oxygen, and glucose, which are used by plants to create their own food and energy.  
\</think\>  
In summary, soil is an incredible resource that supports a wide range of organisms including humans.\<|im\_end|\>  
Aquatic plants have been around for millions of years, with evidence of this long history in the environment. This period also saw the development of various plant species that have adapted to survive in the harsh conditions of their environments. The diverse array of plants found here has made it possible to thrive in different ecosystems over time, providing food, shelter, and resources for human communities.\<|im\_end|\>  
The importance of soil as a vital resource lies in its role as a source of essential nutrients for life on Earth. Soil provides a crucial resource for many different organisms, including humans. However, the presence of soil is not without its challenges. For example, some of these plants may struggle to grow or even die due to lack of sunlight, temperature fluctuations, or other environmental factors. Additionally, the soil's structure can be affected by factors such as water stress, nutrient availability, and pH levels, which can affect plant growth and development in various ecosystems.

\-----------------------------------------------------------------  
  📥 Input Speed (Prefill):    1,847.7 tok/s  (9 tokens in 4.87 ms)  
  📤 Output Speed (Decode):      195.5 tok/s  (300 tokens in 1.53 s)  
  ⏱️  Latency per Token:          5.11 ms/token  
\-----------------------------------------------------------------

\--- Domain: Python Code \---  
def quicksort(arr):  
    """Sorts an array recursively."""  
    def \_\_init\_\_(self, arr):  
        self.arr \= arr

    def recursive\_sorted(self, key, value):  
        if not isinstance(key, dict):  
            return None

    def recursive\_sorted(self, key, value):  
        if not isinstance(value, int):  
            return None

    def recursive\_sorted(self, key, value):  
        for i in range(len(value)):  
            if self.isinstance(value\[i\]):  
                 return self.sorted(value\[i\])

    def recursive\_sorted(self, key, value):  
        for i in range(len(value)):  
            if self.isinstance(value\[i\]):  
                return self.sorted(value\[i\])

    def recursive\_sorted(self, key,  
\-----------------------------------------------------------------  
  📥 Input Speed (Prefill):    3,975.6 tok/s  (22 tokens in 5.53 ms)  
  📤 Output Speed (Decode):      193.1 tok/s  (300 tokens in 1.55 s)  
  ⏱️  Latency per Token:          5.18 ms/token  
\-----------------------------------------------------------------

\> 1024). Running this sequence through the model will result in indexing errors  
SFT Step    25/7500 | Response PPL:   7.57 | CE: 2.0245 | JEPA: 0.4922 | LR: 6.25e-06  
SFT Step    50/7500 | Response PPL:   6.78 | CE: 1.9140 | JEPA: 0.3828 | LR: 1.25e-05  
SFT Step    75/7500 | Response PPL:   6.69 | CE: 1.9000 | JEPA: 0.3653 | LR: 1.88e-05  
SFT Step   100/7500 | Response PPL:   6.50 | CE: 1.8725 | JEPA: 0.3545 | LR: 2.50e-05  
SFT Step   125/7500 | Response PPL:   6.07 | CE: 1.8034 | JEPA: 0.3344 | LR: 3.13e-05  
SFT Step   150/7500 | Response PPL:   6.27 | CE: 1.8356 | JEPA: 0.3390 | LR: 3.75e-05  
SFT Step   175/7500 | Response PPL:   6.20 | CE: 1.8253 | JEPA: 0.3250 | LR: 4.38e-05  
SFT Step   200/7500 | Response PPL:   6.36 | CE: 1.8502 | JEPA: 0.3402 | LR: 5.00e-05  
SFT Step   225/7500 | Response PPL:   5.86 | CE: 1.7678 | JEPA: 0.3405 | LR: 5.00e-05  
SFT Step   250/7500 | Response PPL:   5.81 | CE: 1.7588 | JEPA: 0.3217 | LR: 5.00e-05  
SFT Step   275/7500 | Response PPL:   5.88 | CE: 1.7717 | JEPA: 0.3257 | LR: 5.00e-05  
SFT Step   300/7500 | Response PPL:   5.77 | CE: 1.7520 | JEPA: 0.3199 | LR: 5.00e-05  
SFT Step   325/7500 | Response PPL:   5.69 | CE: 1.7392 | JEPA: 0.3179 | LR: 5.00e-05  
SFT Step   350/7500 | Response PPL:   5.54 | CE: 1.7124 | JEPA: 0.3205 | LR: 5.00e-05  
SFT Step   375/7500 | Response PPL:   5.68 | CE: 1.7376 | JEPA: 0.3234 | LR: 4.99e-05  
SFT Step   400/7500 | Response PPL:   5.66 | CE: 1.7328 | JEPA: 0.3165 | LR: 4.99e-05  
SFT Step   425/7500 | Response PPL:   5.59 | CE: 1.7215 | JEPA: 0.3267 | LR: 4.99e-05  
SFT Step   450/7500 | Response PPL:   5.71 | CE: 1.7415 | JEPA: 0.3127 | LR: 4.99e-05  
SFT Step   475/7500 | Response PPL:   5.45 | CE: 1.6956 | JEPA: 0.3155 | LR: 4.98e-05  
SFT Step   500/7500 | Response PPL:   5.32 | CE: 1.6716 | JEPA: 0.3133 | LR: 4.98e-05  
SFT Step   525/7500 | Response PPL:   5.39 | CE: 1.6846 | JEPA: 0.3162 | LR: 4.98e-05  
SFT Step   550/7500 | Response PPL:   5.12 | CE: 1.6328 | JEPA: 0.3194 | LR: 4.97e-05  
SFT Step   575/7500 | Response PPL:   5.41 | CE: 1.6882 | JEPA: 0.3198 | LR: 4.97e-05  
SFT Step   600/7500 | Response PPL:   5.50 | CE: 1.7051 | JEPA: 0.3202 | LR: 4.97e-05  
SFT Step   625/7500 | Response PPL:   5.24 | CE: 1.6571 | JEPA: 0.3155 | LR: 4.96e-05  
SFT Step   650/7500 | Response PPL:   5.52 | CE: 1.7076 | JEPA: 0.3195 | LR: 4.96e-05  
SFT Step   675/7500 | Response PPL:   5.30 | CE: 1.6671 | JEPA: 0.3095 | LR: 4.95e-05  
SFT Step   700/7500 | Response PPL:   5.31 | CE: 1.6694 | JEPA: 0.3214 | LR: 4.95e-05  
SFT Step   725/7500 | Response PPL:   5.36 | CE: 1.6798 | JEPA: 0.3253 | LR: 4.94e-05  
SFT Step   750/7500 | Response PPL:   5.35 | CE: 1.6763 | JEPA: 0.3151 | LR: 4.94e-05  
SFT Step   775/7500 | Response PPL:   5.37 | CE: 1.6811 | JEPA: 0.3277 | LR: 4.93e-05  
SFT Step   800/7500 | Response PPL:   5.19 | CE: 1.6476 | JEPA: 0.3226 | LR: 4.93e-05  
SFT Step   825/7500 | Response PPL:   5.11 | CE: 1.6309 | JEPA: 0.3128 | LR: 4.92e-05  
SFT Step   850/7500 | Response PPL:   5.23 | CE: 1.6536 | JEPA: 0.3270 | LR: 4.91e-05  
SFT Step   875/7500 | Response PPL:   5.12 | CE: 1.6339 | JEPA: 0.3149 | LR: 4.91e-05  
SFT Step   900/7500 | Response PPL:   5.22 | CE: 1.6517 | JEPA: 0.3211 | LR: 4.90e-05  
SFT Step   925/7500 | Response PPL:   5.09 | CE: 1.6277 | JEPA: 0.3253 | LR: 4.89e-05  
SFT Step   950/7500 | Response PPL:   5.15 | CE: 1.6381 | JEPA: 0.3240 | LR: 4.88e-05  
SFT Step   975/7500 | Response PPL:   5.13 | CE: 1.6344 | JEPA: 0.3253 | LR: 4.88e-05  
SFT Step  1000/7500 | Response PPL:   5.37 | CE: 1.6817 | JEPA: 0.3262 | LR: 4.87e-05  
\--\> Saved SFT Checkpoint: ./checkpoints\_sft/2.0/sft\_step\_1000.pt  
SFT Step  1025/7500 | Response PPL:   5.40 | CE: 1.6871 | JEPA: 0.3282 | LR: 4.86e-05  
SFT Step  1050/7500 | Response PPL:   5.22 | CE: 1.6528 | JEPA: 0.3249 | LR: 4.85e-05  
SFT Step  1075/7500 | Response PPL:   5.08 | CE: 1.6246 | JEPA: 0.3221 | LR: 4.84e-05  
SFT Step  1100/7500 | Response PPL:   5.17 | CE: 1.6427 | JEPA: 0.3304 | LR: 4.83e-05  
SFT Step  1125/7500 | Response PPL:   5.21 | CE: 1.6498 | JEPA: 0.3304 | LR: 4.82e-05  
SFT Step  1150/7500 | Response PPL:   5.40 | CE: 1.6870 | JEPA: 0.3238 | LR: 4.81e-05  
SFT Step  1175/7500 | Response PPL:   5.03 | CE: 1.6152 | JEPA: 0.3179 | LR: 4.81e-05  
SFT Step  1200/7500 | Response PPL:   5.07 | CE: 1.6229 | JEPA: 0.3229 | LR: 4.80e-05  
SFT Step  1225/7500 | Response PPL:   5.26 | CE: 1.6609 | JEPA: 0.3337 | LR: 4.79e-05  
SFT Step  1250/7500 | Response PPL:   5.12 | CE: 1.6337 | JEPA: 0.3213 | LR: 4.77e-05  
SFT Step  1275/7500 | Response PPL:   5.11 | CE: 1.6316 | JEPA: 0.3296 | LR: 4.76e-05  
SFT Step  1300/7500 | Response PPL:   5.23 | CE: 1.6549 | JEPA: 0.3246 | LR: 4.75e-05  
SFT Step  1325/7500 | Response PPL:   4.93 | CE: 1.5949 | JEPA: 0.3274 | LR: 4.74e-05  
SFT Step  1350/7500 | Response PPL:   4.96 | CE: 1.6021 | JEPA: 0.3309 | LR: 4.73e-05  
SFT Step  1375/7500 | Response PPL:   5.15 | CE: 1.6391 | JEPA: 0.3228 | LR: 4.72e-05  
SFT Step  1400/7500 | Response PPL:   5.04 | CE: 1.6175 | JEPA: 0.3210 | LR: 4.71e-05  
SFT Step  1425/7500 | Response PPL:   5.07 | CE: 1.6233 | JEPA: 0.3255 | LR: 4.69e-05  
SFT Step  1450/7500 | Response PPL:   5.24 | CE: 1.6554 | JEPA: 0.3243 | LR: 4.68e-05  
SFT Step  1475/7500 | Response PPL:   5.26 | CE: 1.6610 | JEPA: 0.3207 | LR: 4.67e-05  
SFT Step  1500/7500 | Response PPL:   5.16 | CE: 1.6401 | JEPA: 0.3286 | LR: 4.66e-05  
SFT Step  1525/7500 | Response PPL:   5.05 | CE: 1.6192 | JEPA: 0.3223 | LR: 4.64e-05  
SFT Step  1550/7500 | Response PPL:   5.05 | CE: 1.6196 | JEPA: 0.3225 | LR: 4.63e-05  
SFT Step  1575/7500 | Response PPL:   5.28 | CE: 1.6646 | JEPA: 0.3368 | LR: 4.62e-05  
SFT Step  1600/7500 | Response PPL:   5.07 | CE: 1.6231 | JEPA: 0.3192 | LR: 4.60e-05  
SFT Step  1625/7500 | Response PPL:   5.10 | CE: 1.6288 | JEPA: 0.3225 | LR: 4.59e-05  
SFT Step  1650/7500 | Response PPL:   5.07 | CE: 1.6225 | JEPA: 0.3149 | LR: 4.58e-05  
SFT Step  1675/7500 | Response PPL:   4.89 | CE: 1.5868 | JEPA: 0.3186 | LR: 4.56e-05  
SFT Step  1700/7500 | Response PPL:   4.96 | CE: 1.6013 | JEPA: 0.3098 | LR: 4.55e-05  
SFT Step  1725/7500 | Response PPL:   5.15 | CE: 1.6381 | JEPA: 0.3197 | LR: 4.53e-05  
SFT Step  1750/7500 | Response PPL:   5.00 | CE: 1.6103 | JEPA: 0.3185 | LR: 4.52e-05  
SFT Step  1775/7500 | Response PPL:   5.02 | CE: 1.6135 | JEPA: 0.3201 | LR: 4.50e-05  
SFT Step  1800/7500 | Response PPL:   5.04 | CE: 1.6165 | JEPA: 0.3334 | LR: 4.49e-05  
SFT Step  1825/7500 | Response PPL:   4.85 | CE: 1.5793 | JEPA: 0.3263 | LR: 4.47e-05  
SFT Step  1850/7500 | Response PPL:   4.96 | CE: 1.6014 | JEPA: 0.3284 | LR: 4.46e-05  
SFT Step  1875/7500 | Response PPL:   5.03 | CE: 1.6162 | JEPA: 0.3328 | LR: 4.44e-05  
SFT Step  1900/7500 | Response PPL:   4.98 | CE: 1.6044 | JEPA: 0.3194 | LR: 4.42e-05  
SFT Step  1925/7500 | Response PPL:   5.15 | CE: 1.6398 | JEPA: 0.3255 | LR: 4.41e-05  
SFT Step  1950/7500 | Response PPL:   5.03 | CE: 1.6158 | JEPA: 0.3282 | LR: 4.39e-05  
SFT Step  1975/7500 | Response PPL:   5.11 | CE: 1.6319 | JEPA: 0.3377 | LR: 4.38e-05  
SFT Step  2000/7500 | Response PPL:   4.89 | CE: 1.5862 | JEPA: 0.3174 | LR: 4.36e-05  
\--\> Saved SFT Checkpoint: ./checkpoints\_sft/2.0/sft\_step\_2000.pt  
SFT Step  2025/7500 | Response PPL:   5.13 | CE: 1.6353 | JEPA: 0.3211 | LR: 4.34e-05  
SFT Step  2050/7500 | Response PPL:   4.86 | CE: 1.5801 | JEPA: 0.3316 | LR: 4.32e-05  
SFT Step  2075/7500 | Response PPL:   5.07 | CE: 1.6225 | JEPA: 0.3240 | LR: 4.31e-05  
SFT Step  2100/7500 | Response PPL:   4.91 | CE: 1.5909 | JEPA: 0.3239 | LR: 4.29e-05  
SFT Step  2125/7500 | Response PPL:   4.98 | CE: 1.6051 | JEPA: 0.3214 | LR: 4.27e-05  
SFT Step  2150/7500 | Response PPL:   5.02 | CE: 1.6133 | JEPA: 0.3304 | LR: 4.25e-05  
SFT Step  2175/7500 | Response PPL:   5.01 | CE: 1.6115 | JEPA: 0.3339 | LR: 4.24e-05  
SFT Step  2200/7500 | Response PPL:   4.76 | CE: 1.5601 | JEPA: 0.3123 | LR: 4.22e-05  
SFT Step  2225/7500 | Response PPL:   5.00 | CE: 1.6101 | JEPA: 0.3288 | LR: 4.20e-05  
SFT Step  2250/7500 | Response PPL:   4.96 | CE: 1.6006 | JEPA: 0.3121 | LR: 4.18e-05  
SFT Step  2275/7500 | Response PPL:   4.99 | CE: 1.6081 | JEPA: 0.3233 | LR: 4.16e-05  
SFT Step  2300/7500 | Response PPL:   4.95 | CE: 1.6000 | JEPA: 0.3238 | LR: 4.14e-05  
SFT Step  2325/7500 | Response PPL:   4.95 | CE: 1.5994 | JEPA: 0.3243 | LR: 4.12e-05  
SFT Step  2350/7500 | Response PPL:   5.05 | CE: 1.6187 | JEPA: 0.3254 | LR: 4.10e-05  
SFT Step  2375/7500 | Response PPL:   5.09 | CE: 1.6279 | JEPA: 0.3232 | LR: 4.09e-05  
SFT Step  2400/7500 | Response PPL:   4.97 | CE: 1.6043 | JEPA: 0.3241 | LR: 4.07e-05  
SFT Step  2425/7500 | Response PPL:   4.93 | CE: 1.5955 | JEPA: 0.3207 | LR: 4.05e-05  
SFT Step  2450/7500 | Response PPL:   5.04 | CE: 1.6180 | JEPA: 0.3219 | LR: 4.03e-05  
SFT Step  2475/7500 | Response PPL:   4.83 | CE: 1.5749 | JEPA: 0.3228 | LR: 4.01e-05  
SFT Step  2500/7500 | Response PPL:   5.24 | CE: 1.6561 | JEPA: 0.3301 | LR: 3.99e-05  
SFT Step  2525/7500 | Response PPL:   4.83 | CE: 1.5743 | JEPA: 0.3251 | LR: 3.97e-05  
SFT Step  2550/7500 | Response PPL:   4.92 | CE: 1.5927 | JEPA: 0.3242 | LR: 3.94e-05  
SFT Step  2575/7500 | Response PPL:   4.86 | CE: 1.5819 | JEPA: 0.3333 | LR: 3.92e-05  
SFT Step  2600/7500 | Response PPL:   4.91 | CE: 1.5906 | JEPA: 0.3180 | LR: 3.90e-05  
SFT Step  2625/7500 | Response PPL:   4.98 | CE: 1.6061 | JEPA: 0.3297 | LR: 3.88e-05  
SFT Step  2650/7500 | Response PPL:   4.95 | CE: 1.5988 | JEPA: 0.3169 | LR: 3.86e-05  
SFT Step  2675/7500 | Response PPL:   4.73 | CE: 1.5535 | JEPA: 0.3180 | LR: 3.84e-05  
SFT Step  2700/7500 | Response PPL:   4.83 | CE: 1.5741 | JEPA: 0.3131 | LR: 3.82e-05  
SFT Step  2725/7500 | Response PPL:   4.96 | CE: 1.6015 | JEPA: 0.3341 | LR: 3.80e-05  
SFT Step  2750/7500 | Response PPL:   5.01 | CE: 1.6118 | JEPA: 0.3340 | LR: 3.78e-05  
SFT Step  2775/7500 | Response PPL:   4.76 | CE: 1.5594 | JEPA: 0.3194 | LR: 3.76e-05  
SFT Step  2800/7500 | Response PPL:   5.00 | CE: 1.6098 | JEPA: 0.3270 | LR: 3.73e-05  
SFT Step  2825/7500 | Response PPL:   4.78 | CE: 1.5654 | JEPA: 0.3333 | LR: 3.71e-05  
SFT Step  2850/7500 | Response PPL:   4.77 | CE: 1.5624 | JEPA: 0.3134 | LR: 3.69e-05  
SFT Step  2875/7500 | Response PPL:   4.88 | CE: 1.5861 | JEPA: 0.3148 | LR: 3.67e-05  
SFT Step  2900/7500 | Response PPL:   4.94 | CE: 1.5968 | JEPA: 0.3228 | LR: 3.65e-05  
SFT Step  2925/7500 | Response PPL:   4.80 | CE: 1.5695 | JEPA: 0.3220 | LR: 3.62e-05  
SFT Step  2950/7500 | Response PPL:   5.01 | CE: 1.6105 | JEPA: 0.3230 | LR: 3.60e-05  
SFT Step  2975/7500 | Response PPL:   4.81 | CE: 1.5704 | JEPA: 0.3179 | LR: 3.58e-05  
SFT Step  3000/7500 | Response PPL:   4.99 | CE: 1.6083 | JEPA: 0.3222 | LR: 3.56e-05  
\--\> Saved SFT Checkpoint: ./checkpoints\_sft/2.0/sft\_step\_3000.pt  
SFT Step  3025/7500 | Response PPL:   4.86 | CE: 1.5814 | JEPA: 0.3214 | LR: 3.53e-05  
SFT Step  3050/7500 | Response PPL:   4.96 | CE: 1.6020 | JEPA: 0.3266 | LR: 3.51e-05  
SFT Step  3075/7500 | Response PPL:   4.85 | CE: 1.5793 | JEPA: 0.3235 | LR: 3.49e-05  
SFT Step  3100/7500 | Response PPL:   5.07 | CE: 1.6227 | JEPA: 0.3333 | LR: 3.46e-05  
SFT Step  3125/7500 | Response PPL:   4.92 | CE: 1.5926 | JEPA: 0.3333 | LR: 3.44e-05  
SFT Step  3150/7500 | Response PPL:   4.89 | CE: 1.5870 | JEPA: 0.3225 | LR: 3.42e-05  
SFT Step  3175/7500 | Response PPL:   4.89 | CE: 1.5869 | JEPA: 0.3260 | LR: 3.40e-05  
SFT Step  3200/7500 | Response PPL:   4.69 | CE: 1.5449 | JEPA: 0.3174 | LR: 3.37e-05  
SFT Step  3225/7500 | Response PPL:   4.78 | CE: 1.5638 | JEPA: 0.3276 | LR: 3.35e-05  
SFT Step  3250/7500 | Response PPL:   5.03 | CE: 1.6147 | JEPA: 0.3304 | LR: 3.33e-05  
SFT Step  3275/7500 | Response PPL:   5.13 | CE: 1.6360 | JEPA: 0.3339 | LR: 3.30e-05  
SFT Step  3300/7500 | Response PPL:   4.74 | CE: 1.5553 | JEPA: 0.3231 | LR: 3.28e-05  
SFT Step  3325/7500 | Response PPL:   4.80 | CE: 1.5679 | JEPA: 0.3284 | LR: 3.25e-05  
SFT Step  3350/7500 | Response PPL:   4.65 | CE: 1.5363 | JEPA: 0.3239 | LR: 3.23e-05  
SFT Step  3375/7500 | Response PPL:   4.74 | CE: 1.5565 | JEPA: 0.3270 | LR: 3.21e-05  
SFT Step  3400/7500 | Response PPL:   4.75 | CE: 1.5584 | JEPA: 0.3234 | LR: 3.18e-05  
SFT Step  3425/7500 | Response PPL:   4.67 | CE: 1.5422 | JEPA: 0.3132 | LR: 3.16e-05  
SFT Step  3450/7500 | Response PPL:   4.79 | CE: 1.5657 | JEPA: 0.3082 | LR: 3.14e-05  
SFT Step  3475/7500 | Response PPL:   5.01 | CE: 1.6109 | JEPA: 0.3264 | LR: 3.11e-05  
SFT Step  3500/7500 | Response PPL:   4.77 | CE: 1.5625 | JEPA: 0.3240 | LR: 3.09e-05  
SFT Step  3525/7500 | Response PPL:   4.79 | CE: 1.5661 | JEPA: 0.3311 | LR: 3.06e-05  
SFT Step  3550/7500 | Response PPL:   4.81 | CE: 1.5707 | JEPA: 0.3231 | LR: 3.04e-05  
SFT Step  3575/7500 | Response PPL:   4.86 | CE: 1.5802 | JEPA: 0.3213 | LR: 3.02e-05  
SFT Step  3600/7500 | Response PPL:   4.84 | CE: 1.5772 | JEPA: 0.3192 | LR: 2.99e-05  
SFT Step  3625/7500 | Response PPL:   4.73 | CE: 1.5541 | JEPA: 0.3269 | LR: 2.97e-05  
SFT Step  3650/7500 | Response PPL:   4.97 | CE: 1.6039 | JEPA: 0.3284 | LR: 2.94e-05  
SFT Step  3675/7500 | Response PPL:   4.81 | CE: 1.5713 | JEPA: 0.3142 | LR: 2.92e-05  
SFT Step  3700/7500 | Response PPL:   5.08 | CE: 1.6255 | JEPA: 0.3251 | LR: 2.90e-05  
SFT Step  3725/7500 | Response PPL:   4.69 | CE: 1.5462 | JEPA: 0.3260 | LR: 2.87e-05  
SFT Step  3750/7500 | Response PPL:   4.99 | CE: 1.6072 | JEPA: 0.3341 | LR: 2.85e-05  
SFT Step  3775/7500 | Response PPL:   4.99 | CE: 1.6077 | JEPA: 0.3290 | LR: 2.82e-05  
SFT Step  3800/7500 | Response PPL:   4.59 | CE: 1.5237 | JEPA: 0.3156 | LR: 2.80e-05  
SFT Step  3825/7500 | Response PPL:   4.59 | CE: 1.5228 | JEPA: 0.3210 | LR: 2.78e-05  
SFT Step  3850/7500 | Response PPL:   4.91 | CE: 1.5910 | JEPA: 0.3358 | LR: 2.75e-05  
SFT Step  3875/7500 | Response PPL:   4.63 | CE: 1.5319 | JEPA: 0.3268 | LR: 2.73e-05  
SFT Step  3900/7500 | Response PPL:   4.80 | CE: 1.5676 | JEPA: 0.3329 | LR: 2.70e-05  
SFT Step  3925/7500 | Response PPL:   4.79 | CE: 1.5660 | JEPA: 0.3261 | LR: 2.68e-05  
SFT Step  3950/7500 | Response PPL:   4.75 | CE: 1.5586 | JEPA: 0.3416 | LR: 2.65e-05  
SFT Step  3975/7500 | Response PPL:   4.79 | CE: 1.5662 | JEPA: 0.3234 | LR: 2.63e-05  
SFT Step  4000/7500 | Response PPL:   4.80 | CE: 1.5694 | JEPA: 0.3395 | LR: 2.61e-05  
\--\> Saved SFT Checkpoint: ./checkpoints\_sft/2.0/sft\_step\_4000.pt  
SFT Step  4025/7500 | Response PPL:   4.90 | CE: 1.5897 | JEPA: 0.3355 | LR: 2.58e-05  
SFT Step  4050/7500 | Response PPL:   4.72 | CE: 1.5525 | JEPA: 0.3183 | LR: 2.56e-05  
SFT Step  4075/7500 | Response PPL:   4.80 | CE: 1.5687 | JEPA: 0.3296 | LR: 2.53e-05  
SFT Step  4100/7500 | Response PPL:   4.79 | CE: 1.5664 | JEPA: 0.3240 | LR: 2.51e-05  
SFT Step  4125/7500 | Response PPL:   4.69 | CE: 1.5445 | JEPA: 0.3305 | LR: 2.49e-05  
SFT Step  4150/7500 | Response PPL:   4.78 | CE: 1.5653 | JEPA: 0.3257 | LR: 2.46e-05  
SFT Step  4175/7500 | Response PPL:   4.77 | CE: 1.5627 | JEPA: 0.3316 | LR: 2.44e-05  
SFT Step  4200/7500 | Response PPL:   4.64 | CE: 1.5342 | JEPA: 0.3263 | LR: 2.41e-05  
SFT Step  4225/7500 | Response PPL:   4.74 | CE: 1.5558 | JEPA: 0.3309 | LR: 2.39e-05  
SFT Step  4250/7500 | Response PPL:   4.78 | CE: 1.5644 | JEPA: 0.3185 | LR: 2.37e-05  
SFT Step  4275/7500 | Response PPL:   4.85 | CE: 1.5780 | JEPA: 0.3246 | LR: 2.34e-05  
SFT Step  4300/7500 | Response PPL:   4.69 | CE: 1.5460 | JEPA: 0.3225 | LR: 2.32e-05  
SFT Step  4325/7500 | Response PPL:   4.72 | CE: 1.5523 | JEPA: 0.3184 | LR: 2.29e-05  
SFT Step  4350/7500 | Response PPL:   4.73 | CE: 1.5548 | JEPA: 0.3215 | LR: 2.27e-05  
SFT Step  4375/7500 | Response PPL:   4.69 | CE: 1.5455 | JEPA: 0.3239 | LR: 2.25e-05  
SFT Step  4400/7500 | Response PPL:   4.85 | CE: 1.5799 | JEPA: 0.3241 | LR: 2.22e-05  
SFT Step  4425/7500 | Response PPL:   4.74 | CE: 1.5564 | JEPA: 0.3078 | LR: 2.20e-05  
SFT Step  4450/7500 | Response PPL:   4.62 | CE: 1.5302 | JEPA: 0.3202 | LR: 2.18e-05  
SFT Step  4475/7500 | Response PPL:   4.61 | CE: 1.5281 | JEPA: 0.3213 | LR: 2.15e-05  
SFT Step  4500/7500 | Response PPL:   4.80 | CE: 1.5696 | JEPA: 0.3239 | LR: 2.13e-05  
SFT Step  4525/7500 | Response PPL:   4.69 | CE: 1.5447 | JEPA: 0.3216 | LR: 2.11e-05  
SFT Step  4550/7500 | Response PPL:   4.70 | CE: 1.5484 | JEPA: 0.3362 | LR: 2.08e-05  
SFT Step  4575/7500 | Response PPL:   4.68 | CE: 1.5438 | JEPA: 0.3282 | LR: 2.06e-05  
SFT Step  4600/7500 | Response PPL:   4.67 | CE: 1.5406 | JEPA: 0.3205 | LR: 2.04e-05

x **.2 lr for bayesian centroids \- kinda garbo**

\=================================================================  
🧠 BAYESIAN CENTROID MEMORY DIAGNOSTICS  
\=================================================================  
Total Prototypes:         512  
Average Variance (σ²):    1.0089  (Prior \= 1.0000)  
Min Variance (Sharpest):  0.7294  
Max Variance (Diffuse):   1.3032  
Prototype Norm Mean:      1.3423  
Active Memory Spread:     0.4379

\=================================================================  
🗺️ COGNITIVE MAP & JEPA MECHANISM PROBE  
\=================================================================  
Cognitive Map Step Velocity (Avg): 3.2676  
Total Manifold Path Displacement:   41.1749  
Centroids Activated by Prompt:      9/512 unique clusters  
JEPA Next-Latent Cosine Sim:        0.6318  (63.2% alignment)

\=================================================================  
⚡ GENERATIVE PROBES & HARDWARE THROUGHPUT BENCHMARKS  
Sampling: Temp=0.65 | Top-P=0.85 | Top-K=40 | Rep-Penalty=1.2  
\=================================================================

\--- Domain: Web/Educational \---  
The theory of general relativity explains that gravity is not a force, but rather an curvature of spacetime caused by the curvature of spacetime.

To understand this concept better, let's consider two possible scenarios: the mass-energy scenario and the gravitational wave scenario. The mass-energy scenario describes how massive objects can move in space, while the gravitational wave scenario describes how strong particles can interact with matter and cause gravitational waves to propagate away from their source.

One way to approach this is through the lens of quantum mechanics. According to classical physics, gravity works as a force between particles, which means that it doesn't affect them directly. However, according to quantum mechanics, gravity does not have any effect on the particles themselves; instead, it affects the behavior of these particles. For example, if an object is moving at high speeds, its motion will be slower than if it were moving at very fast speeds, but the object's mass would be more dense and difficult to move. This creates a kind of "gravimetric" phenomenon where objects can travel in parallel or even across vast distances, regardless of their mass-energy state.

Another approach is to use the concept of curvature to describe the behavior of spacetime. The curvature of spacetime refers to how much matter and energy are curved around a central point, which means that there is no net force acting on them. In this case, the object is still moving in a normal gravitational field, so its motion will be slower than if it  
\-----------------------------------------------------------------  
  📥 Input Speed (Prefill):      203.1 tok/s  (9 tokens in 44.31 ms)  
  📤 Output Speed (Decode):       26.1 tok/s  (300 tokens in 11.51 s)  
  ⏱️  Latency per Token:         38.38 ms/token  
\-----------------------------------------------------------------

\--- Domain: Synthetic Reasoning \---  
Photosynthesis is crucial for life on Earth because it helps to convert light energy into chemical energy. The process of photosynthesis involves the breakdown of glucose and oxygen, which are two primary components that provide the building blocks for all organisms.

The process of photosynthesis is a complex biological process where plant cells absorb sunlight from their leaves and use it as fuel to produce energy. During this process, plants absorb carbon dioxide from the air through their roots and use water to grow and multiply. This process also provides energy for growth and development, as the process also plays a role in maintaining the environment and supporting the survival of species.

In conclusion, photosynthesis is an essential biological process that plays a crucial role in the life cycle of plants and animals. It is a process that allows plants to convert light energy into chemical energy, which can be used by plants to drive growth, develop new traits, and sustain themselves on the planet. The process is a vital aspect of the ecosystem and has numerous applications across various industries.\<|im\_end|\>

\*\*The Life Cycle: What Is It?\*\*

At its core, the life cycle of a plant or animal is one of continuous changes in their internal and external environment over time. This is achieved through several key stages, including the plant's root system (where it provides nutrients for growth), the plant's root system (where it supports growth and development) and the plant's leaves (where they provide energy).

As the plant begins to grow and multiply, it undergo  
\-----------------------------------------------------------------  
  📥 Input Speed (Prefill):      979.5 tok/s  (9 tokens in 9.19 ms)  
  📤 Output Speed (Decode):       52.0 tok/s  (300 tokens in 5.77 s)  
  ⏱️  Latency per Token:         19.24 ms/token  
\-----------------------------------------------------------------

\--- Domain: Python Code \---  
def quicksort(arr):  
    """Sorts an array recursively."""  
    def \_\_init\_\_(self, start=0) \-\> int:  
        self.start \= start  
        self.end \= end  
\`\`\`\<|im\_end|\>

The code above is a recursive implementation of the \`quicksort\` function in Python. It initializes two variables, \`start\`, which represents the starting index and the number of elements in each array, respectively, to 0\. The function then iterates through these arrays using a for loop, checking if the current element is already in the \`start\`. If it is not already in the \`start\`, it increments the \`start\` variable. Finally, it returns the current value of the \`start\` variable.\<|im\_end|\>

Here's how you could do it:

\`\`\`python  
def quicksort(arr):  
    """Sorts an array recursively."""  
    def \_\_init\_\_(self, start=0) \-\> int:  
        self.start \= start  
        self.end \= end

    def is\_empty(self, end=0):  
        if not self.start or len(arr) \!= len(arr\[start\]) or len(arr\[0\] \== len(arr\[1\]))  
\-----------------------------------------------------------------  
  📥 Input Speed (Prefill):    4,657.5 tok/s  (22 tokens in 4.72 ms)  
  📤 Output Speed (Decode):       93.9 tok/s  (300 tokens in 3.19 s)  
  ⏱️  Latency per Token:         10.65 ms/token  
\-----------------------------------------------------------------

**X .05 lr for bayesian centroids on fine tuning for 1000 steps, cosine similarity on original dataset down by 10%, coding seemed bad, other domains I’m not even sure about honestly**

Loading checkpoint: ./checkpoints\_sft/3.0/sft\_step\_1000.pt ...  
✅ Successfully loaded model from Step 1000\!

\=================================================================  
🧠 BAYESIAN CENTROID MEMORY DIAGNOSTICS  
\=================================================================  
Total Prototypes:         512  
Average Variance (σ²):    1.0090  (Prior \= 1.0000)  
Min Variance (Sharpest):  0.7276  
Max Variance (Diffuse):   1.3039  
Prototype Norm Mean:      1.3409  
Active Memory Spread:     0.4377

\=================================================================  
🗺️ COGNITIVE MAP & JEPA MECHANISM PROBE  
\=================================================================  
Cognitive Map Step Velocity (Avg): 3.2670  
Total Manifold Path Displacement:   40.8436  
Centroids Activated by Prompt:      8/512 unique clusters  
JEPA Next-Latent Cosine Sim:        0.6534  (65.3% alignment)

\=================================================================  
⚡ GENERATIVE PROBES & HARDWARE THROUGHPUT BENCHMARKS  
Sampling: Temp=0.65 | Top-P=0.85 | Top-K=40 | Rep-Penalty=1.2  
\=================================================================

\--- Domain: Web/Educational \---  
The theory of general relativity explains that gravity is a force that is not the same as it is in other dimensions, such as space and time.

One way to understand this concept is to consider the behavior of objects in spacetime around a central point called the LZ. The LZ is a curved manifold with a curvature that can be described by its curvature using a mathematical formula:

L \= m \* h  
where m is the mass of the object (in this case, the LZ), and h is the gravitational constant. This equation describes how much matter is in a particular region of spacetime.

To illustrate this, let's imagine a scenario where two identical twins are sitting at different positions on the same plane. They are both moving through space, but their movements are different due to their respective gravitational forces. According to the theory of general relativity, they would experience a similar state when they move through spacetime around the LZ.

In this scenario, the gravitational force is proportional to the distance between them, which is given by:

h \= m \* h  
where h is the distance from one point to another (in this case, the LZ). This relationship can be understood using the concept of gravitational fields and the curvature of spacetime. The distance between the two objects is denoted as s^2 or s^n, where s is the distance between them.

Now, let's consider how the gravitational field interacts with the matter in question. Since  
\-----------------------------------------------------------------  
  📥 Input Speed (Prefill):      211.7 tok/s  (9 tokens in 42.50 ms)  
  📤 Output Speed (Decode):       29.9 tok/s  (300 tokens in 10.03 s)  
  ⏱️  Latency per Token:         33.44 ms/token  
\-----------------------------------------------------------------

\--- Domain: Synthetic Reasoning \---  
Photosynthesis is crucial for life on Earth because it helps plants to convert light energy into chemical energy, and the process of photosynthesis involves the transfer of water from the atmosphere to the plants.”  
The team discovered that the key to this process was a special type of bacteria called anaerobic bacteria. These bacteria are not only responsible for breaking down organic matter but also produce oxygen through a process known as electron transfer. The team found that these bacteria have a unique ability to absorb and process hydrogen ions, which can be used in various applications like fuel cells, nuclear power, and even water treatment systems.

This discovery highlights the importance of understanding and harnessing the power of nature for sustainable energy production and the potential to create new technologies to improve human health and environmental sustainability.\<|im\_end|\>

\*\*What makes the research particularly interesting is the way it challenges traditional views on energy production from fossil fuels."\<|im\_end|\>

The study of natural processes has led to significant advances in our understanding of how energy can be generated using renewable resources such as solar, wind, and geothermal energy. For instance, researchers have shown that these sources of electricity can be harnessed through the use of advanced nanotechnology techniques like nanostructured catalysts or advanced metallization methods. These processes are not only sustainable but also provide a unique opportunity for innovation in energy production technologies.

In addition, the team discovered that the bacteria themselves can be used to create novel materials with exceptional properties, including flexible  
\-----------------------------------------------------------------  
  📥 Input Speed (Prefill):    1,865.3 tok/s  (9 tokens in 4.82 ms)  
  📤 Output Speed (Decode):       95.0 tok/s  (300 tokens in 3.16 s)  
  ⏱️  Latency per Token:         10.53 ms/token  
\-----------------------------------------------------------------

\--- Domain: Python Code \---  
def quicksort(arr):  
    """Sorts an array recursively."""  
    for i in range(len(arr) \- 1, len(arr)):  
        if arr\[i\] \== 0 and arr\[i\] \== 0:  
           return None

    return \[result.to\_dict() for arr in arr\]  
\`\`\`

This function uses the \`ascii.solve\` method to find the solution to a quadratic equation using the \`scipy.integrate\` function from the \`math\` module. The \`ascii.solve\` function is used to solve the quadratic equation, which is often found in the problem solving quadratic equations.\<|im\_end|\>

The function returns the result of the quadratic equation as a tuple of the solutions to the quadratic equation.\<|im\_end|\>

Here's how you can use this function:

\`\`\`python  
def quadr\_calculator(arr):  
    """Sorts an array recursively."""  
    return arr\[0\] \* (arr\*\*2 \+ 1\) for arr in arr

    return \[result.to\_dict() for arr in arr\]  
\`\`\`\<|im\_end|\>

This implementation uses the \`ascii.solve\` method to find the solution to a  
\-----------------------------------------------------------------  
  📥 Input Speed (Prefill):    5,176.2 tok/s  (22 tokens in 4.25 ms)  
  📤 Output Speed (Decode):       65.2 tok/s  (300 tokens in 4.60 s)  
  ⏱️  Latency per Token:         15.34 ms/token  
\-----------------------------------------------------------------

**Much worse output overall, 9 centroids activated and jepa cosine similarity dropped 1%**  
✅ Successfully loaded model from Step 2000\!

\=================================================================  
🧠 BAYESIAN CENTROID MEMORY DIAGNOSTICS  
\=================================================================  
Total Prototypes:         512  
Average Variance (σ²):    1.0089  (Prior \= 1.0000)  
Min Variance (Sharpest):  0.7281  
Max Variance (Diffuse):   1.3039  
Prototype Norm Mean:      1.3421  
Active Memory Spread:     0.4374

\=================================================================  
🗺️ COGNITIVE MAP & JEPA MECHANISM PROBE  
\=================================================================  
Cognitive Map Step Velocity (Avg): 3.4834  
Total Manifold Path Displacement:   44.5416  
Centroids Activated by Prompt:      9/512 unique clusters  
JEPA Next-Latent Cosine Sim:        0.6465  (64.6% alignment)

\=================================================================  
⚡ GENERATIVE PROBES & HARDWARE THROUGHPUT BENCHMARKS  
Sampling: Temp=0.65 | Top-P=0.85 | Top-K=40 | Rep-Penalty=1.2  
\=================================================================

\--- Domain: Web/Educational \---  
The theory of general relativity explains that gravity is a force between two massive objects, which means that it can be pulled by an object in the same direction. The concept of gravitational waves has been studied for centuries and continues to influence our understanding of the universe.

In conclusion, while gravity plays a significant role in the fabric of space-time, its effects are complex and multifaceted. By examining the laws of physics and the principles underlying them, we can gain insights into the fundamental forces at play in the universe.\<|im\_end|\>

From a purely theoretical perspective, gravity's impact on space-time is profound and far-reaching. It is not just a physical force but also a manifestation of quantum mechanics, where it affects every aspect of the universe. This is why Einstein's theory of relativity is considered one of the most famous theories in physics today, despite its limitations.

The relationship between gravity and space-time is a subject of ongoing research, as scientists continue to explore new questions about the nature of gravity and the universe itself. The theory of general relativity, which describes how objects move under gravity, has been extensively studied for centuries, with some arguing that it provides an explanation for the observed behavior. However, this theory remains a topic of ongoing debate among physicists due to its complex and multifaceted nature.

In conclusion, while gravity is a fundamental force that shapes our experience, its effects are not limited by classical mechanics alone. The theory of general relativity, which describes how objects interact with  
\-----------------------------------------------------------------  
  📥 Input Speed (Prefill):      181.6 tok/s  (9 tokens in 49.56 ms)  
  📤 Output Speed (Decode):       27.2 tok/s  (300 tokens in 11.03 s)  
  ⏱️  Latency per Token:         36.77 ms/token  
\-----------------------------------------------------------------

\--- Domain: Synthetic Reasoning \---  
Photosynthesis is crucial for life on Earth because it helps to convert sunlight into energy and oxygen.  
The process of photosynthesis is called photosynthesis, which is a vital process that involves the transfer of water from one part of the plant's root system to another, in this case, the leaves of the plant or the roots of other plants.\<|im\_end|\>  
This process can be described as an intricate dance between light and darkness, where the plants receive and store their food in a way that allows them to grow and thrive without any external energy sources. The process begins with light and then moves through the air, where it's converted into glucose, a type of sugar that the plants use for growth and energy.\<|im\_end|\>

\*\*The Soil and Water\*\*

The first step in photosynthesis is the transformation of sunlight into glucose, which is a crucial nutrient for the plants. This process occurs when the plant receives light from the sun or other sources, such as a sunny window or under a warm blanket, and stores its energy in the form of oxygen. The glucose molecules are split open and release energy in the form of ATP (adenosine triphosphate), which is used to power the plant's cells. The glucose molecules then travel through the air and water, where they undergo a chemical reaction that breaks down the glucose into glucose and oxygen.\<|im\_end|\>

\*\*Lighting and Water\*\*

The light-dark cycle is essential for photosynthesis because  
\-----------------------------------------------------------------  
  📥 Input Speed (Prefill):    1,912.9 tok/s  (9 tokens in 4.70 ms)  
  📤 Output Speed (Decode):       52.6 tok/s  (300 tokens in 5.70 s)  
  ⏱️  Latency per Token:         18.99 ms/token  
\-----------------------------------------------------------------

\--- Domain: Python Code \---  
def quicksort(arr):  
    """Sorts an array recursively."""  
    for i in range(len(arr) \- 1):  
        if arr\[i\] \< 0 and arr\[i \+ 1\] \== arr\[i+1\]:  
            return arr\[i\], arr\[i+1\].values()  
\`\`\`\<|im\_end|\>

This problem can be solved using a recursive approach. The recursive method is used to iterate through the elements of the array, finding the next element that is the most common value between them. This process can be repeated until no more elements are found. To solve this problem, you can use a simple loop or an empty array.

Here's how you can implement it:

\`\`\`python  
def recursively\_sorted(arr):  
    """Sorts an array recursively."""  
    arr \= \[\]  
    for i in range(len(arr) \- 1):  
        if arr\[i\] \< 0 and arr\[i \+ 1\] \== arr\[i+1\]:  
           arr.append(arr\[i\])

    return arr  
\`\`\`\<|im\_end|\>

**Still worse output at x .05 lr x.1 clearly better in this case**

✅ Successfully loaded model from Step 3000\!

\=================================================================  
🧠 BAYESIAN CENTROID MEMORY DIAGNOSTICS  
\=================================================================  
Total Prototypes:         512  
Average Variance (σ²):    1.0089  (Prior \= 1.0000)  
Min Variance (Sharpest):  0.7287  
Max Variance (Diffuse):   1.3040  
Prototype Norm Mean:      1.3426  
Active Memory Spread:     0.4376

\=================================================================  
🗺️ COGNITIVE MAP & JEPA MECHANISM PROBE  
\=================================================================  
Cognitive Map Step Velocity (Avg): 3.4930  
Total Manifold Path Displacement:   44.6064  
Centroids Activated by Prompt:      9/512 unique clusters  
JEPA Next-Latent Cosine Sim:        0.6366  (63.7% alignment)

\=================================================================  
⚡ GENERATIVE PROBES & HARDWARE THROUGHPUT BENCHMARKS  
Sampling: Temp=0.65 | Top-P=0.85 | Top-K=40 | Rep-Penalty=1.2  
\=================================================================

\--- Domain: Web/Educational \---  
The theory of general relativity explains that gravity is a force between objects, and it's the same as our understanding of gravity.

In this scenario, gravity is not just about how massive objects are moving, but also about what happens to the matter in front of them. In fact, gravity is an infinite force because it's the same as the forces on the Earth.

To see why gravity works, let's consider two scenarios: one where mass is present at the time of the event (in this case, the gravitational pull), and another where gravity is simply the same as the forces on the Earth. The problem of gravity is a fundamental concept in physics, and its explanation involves both the laws of physics and the concepts of general relativity.\<|im\_end|\>

Imagine that you're standing at a distance from a massive object like the Earth, and your goal is to find out what happens to the matter in front of them. To do this, you could place yourself in a position where the force on your body is zero. However, this position doesn't change the way you move around, so it's not directly affected by gravity.

You would need to create an environment that allows for mass transfer or change in direction, such as in the case of gravitational waves. This means that objects moving through space are always moving relative to one another, regardless of their distance between them. If we were to create a gravitational wave experiment, which involves placing a small object near a massive object, and observing its motion  
\-----------------------------------------------------------------  
  📥 Input Speed (Prefill):      184.0 tok/s  (9 tokens in 48.91 ms)  
  📤 Output Speed (Decode):       29.2 tok/s  (300 tokens in 10.29 s)  
  ⏱️  Latency per Token:         34.29 ms/token  
\-----------------------------------------------------------------

\--- Domain: Synthetic Reasoning \---  
Photosynthesis is crucial for life on Earth because it provides the necessary energy to convert sunlight into chemical energy.

The sun's energy can be split in two ways: through solar radiation, which we use as a source of energy or by direct current, and an electric field, which we can harness from wind and other sources like wind turbines. The process involves several steps that involve different components working together to produce energy.

1\. \*\*Solar Energy\*\*: This energy comes from the sun itself. The sun's energy is created when the atmosphere cools down due to the heat generated during the day, but also from the sun's light.  
2\. \*\*Direct Current (DC)\*\*: This current is produced when electricity is drawn from a battery or a substation using alternating current (AC). It generates power in the form of electrical energy.  
3\. \*\*electricity\*\*: This energy comes from the sun's energy and can be harnessed through solar panels, wind turbines, or other forms of renewable energy sources like solar or wind turbines.  
4\. \*\*Wattage\*\*: This energy is used to generate power on a long-term basis by converting sunlight into chemical energy.  
5\. \*\*Solar Power\*\*: This energy is generated using photovoltaic cells or solar collectors that convert light into electrical energy.  
6\. \*\*Solar Power\*\*: This energy is generated using solar panels, which are typically powered by electricity generated from the sun itself.  
7\. \*\*Electrical Energy\*\*:  
\-----------------------------------------------------------------  
  📥 Input Speed (Prefill):    2,147.6 tok/s  (9 tokens in 4.19 ms)  
  📤 Output Speed (Decode):       80.0 tok/s  (300 tokens in 3.75 s)  
  ⏱️  Latency per Token:         12.50 ms/token  
\-----------------------------------------------------------------

\--- Domain: Python Code \---  
def quicksort(arr):  
    """Sorts an array recursively."""  
    arr\[0\] \= 1, arr\[1\] \= 2, arr\[2\] \= 3

    for i in range(len(arr) \- 1):  
        if arr\[i\]\[j\] \> arr\[i+1\]:  
            return arr\[i \+ 1\], arr\[j \+ 1\].sum()  
        else:  
            return arr\[i \+ 1\], arr\[j \+ 1\].sum()  
\`\`\`\<|im\_end|\>

To implement the quicksort algorithm, we need to iterate through each element of the array and check if it's already in the current position. If it's not, we add its value to the result using a for loop. Finally, we calculate the sum of the elements in the current position and subtract the sum from 2 if it's already in the current position.\<|im\_end|\>

Here is how you could do it:

\`\`\`python  
def quicksort(arr):  
    """Sorts an array recursively."""  
    arr\[0\] \= 1, arr\[1\].sum()  
    arr\[2\] \= 3

    for i in range(len(arr) \- 1):

\-----------------------------------------------------------------  
  📥 Input Speed (Prefill):    4,542.4 tok/s  (22 tokens in 4.84 ms)  
  📤 Output Speed (Decode):       63.0 tok/s  (300 tokens in 4.76 s)  
  ⏱️  Latency per Token:         15.86 ms/token  
\-----------------------------------------------------------------

