##### <mark style="background: #D2B3FFA6;">Getting Started on Modeling</mark>
1. Literature search to see what's possible (courses, blogs, open-source projects).
2. If goal is to build something practical and not research based, don't obsess over something new or too good to be true/ greatest. Instead, spend half a day on blog posts and pick something reasonable that lets you get started quickly. Eg., find an open-source implementation that will also help in establishing baseline more efficiently. 
3. For many practical applications, a reasonable algorithm with good data will often do just fine and will in fact outperform a great algorithm with not so good data.

##### <mark style="background: #D2B3FFA6;">Deployment Constraints when picking a model</mark>
Should you take into account deployment constraints when picking a model?
- Yes, if the baseline is already established and goal is to build and deploy.
- No (or not necessarily), if purpose is to establish a baseline and determine what is possible and might be worth pursuing.

##### <mark style="background: #D2B3FFA6;">Sanity-Check for Code and Algorithm</mark>
- Try to overfit a small training dataset before training on a large one (saves time and energy if model isn't working).