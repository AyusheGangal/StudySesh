<mark style="background: #ADCCFFA6;">For a model-centric view</mark>
- You take the data you have and develop a model that does as well as possible on it. 
- We take a benchmark dataset and try to do well on it.
- Most academic research on AI is model-centric.
- Hold the data fixed and iteratively improve the code/model.

<mark style="background: #ADCCFFA6;">For a data-centric view</mark>
- The quality of the data is paramount.
- Use tools (eg, error analysis, data augmentation) to improve the data quality, which will allow multiple models to do well.
- Hold the code fixed and iteratively improve the data.

### <mark style="background: #D2B3FFA6;">Data Augmentation</mark>
Data augmentation can be a very efficient way to get more data, especially for unstructured data problems such as images, audio, maybe text.

<mark style="background: #ADCCFFA6;">Goal: To create a examples the learning algorithm can learn from</mark>

For the <mark style="background: #ABF7F7A6;">speech recognition example</mark>, the decisions which you need to make are:
1. What types of background noise should you use?
2. How loud the background noise should be relative to the speech?

For a systematic way of doing this, Create realistic examples:
- the algorithm does poorly on
- but humans (baseline) do better on

<mark style="background: #ADCCFFA6;">Checklist for generating new data using data augmentation:</mark>
1. Does it sound realistic?
2. Is $X \rightarrow Y$ mapping clear? (ie, are humans able to recognize what was said?)
3. Is the algorithm currently doing poorly on it?

For data augmentation on the <mark style="background: #ABF7F7A6;">image example</mark> (the broken phone screen), 
1. Flip the image horizontally
2. Implement contrast changes (brightening or darkening the image)
3. Use photoshop to create new unique examples
4. Advanced techniques: use GANS to synthesize scratches automatically (could be an overkill too)

### <mark style="background: #D2B3FFA6;">Data Iteration Loop</mark>
- In a data-centric approach AI development, sometimes it's useful to use a data iteration loop where you repeatedly take the data and the model, train your learning algorithm, do error analysis, and as you go through this loop, focus on how to add data or improve the quality of the data. 
- For many practical applications, we perform a robust hyper parameter search with this data iteration loop approach. This results in faster improvements to your learning algorithm performance, depending on your problem. ![[Screenshot 2023-01-14 at 9.04.10 PM.png|500]]
- When working on an unstructured data problem, using data augmentation to create new data that seems realistic and on which humans can do quite well on, but the algorithm struggles on, can be an efficient way to improve the learning algorithm's performance.

### <mark style="background: #D2B3FFA6;">Can Adding More Data Hurt?</mark>
For unstructured data problems, adding data rarely hurts accuracy if:
- The model is large (low bias)
- The mapping $X \rightarrow Y$ is clear (eg, given )

Example where adding data can hurt performance is "1 vs I(i)" when detecting 1's in house numbers.

>[! Up Next]
>[[Data-Centric AI Development for Structured Data]]
