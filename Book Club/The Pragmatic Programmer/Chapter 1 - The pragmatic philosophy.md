### It's your Life
- Make intelligent compromises and informed decisions
- You have agency. It is your life, your decisions. Do not resist change. Don't like what you see, how you feel, you have the ability to change it. "You can change your organization or change your organization".
- The software industry gives your remarkable opportunities, be proactive and take them.

### The cat ate my Source Code
- Take responsibility for yourself and your actions in terms of career advancements, learning, and education.
- Do not be afraid to admit ignorance or error. Be honest and direct about it. We can be proud of our abilities, but we must own up to our shortcomings, ignorances, and mistakes.

#### Take Responsibility (page 4)
- Responsibility is something you actively agree to.
- You have the right to not take on a responsibility, but when you do accept the responsibility of an outcome, you should expect to be held accountable for it. 
- It is up to you to provide solutions, not excuses. Instead of excuses, provide options.
- Sometimes you know what they're going to say, so save them the trouble.

### Software Entropy
- Software rot and tech debt are synonyms. It is a more optimistic term, implying that it will be paid back one day.
- Hopelessness can be contagious. 
- Ignoring a clearly broken situation reinforces the ideas that perhaps nothing can be fixed, no one cares, all is doomed. 
- Don't leave broken windows (bad design, wrong decisions, or poorly written code) unrepaired. Fix each one as soon as it is discovered.
- Neglect accelerates the rot faster than any other factor.

#### First, do no harm
- Do not cause collateral damage just because there is a crisis of some sort. One broken window is one too many.

### Stone soup and Boiled frogs
- People find it easier to join an ongoing success. Show them the glimpse of the future, and you can get them to rally around. 
- Be a catalyst for change.
- Most software disasters start out too small to notice, and most project overruns happen a day at a time. 
- Its often the accumulation of small things that break morale.
- Keep an eye on the big picture. Constantly review what's happening around you, not just what you personally are doing.

### Good enough software
- Users should be given an opportunity to participate in the process of deciding when what you've produced is good enough for their needs.
- The scope and quality of the system you produce should be discussed as part of that system's requirements. Make quality a requirements issue.
- Many users would rather have a software rough around the edges *now*, than to wait a year for a shiny, bells and whistle software. Their needs may change anyway in a year.
- Also if they get the software early, their feedback might lead to a better eventual solution.



## Things I like:
- Like the writing style, it is very informal. Almost like talking to a friendly mentor giving you advice on how to be a good programmer which doubles as life advice.

## Challenges:
1. consider the effect of modularization on the delivery of software. will it take more or less time for a tightly coupled monolithic software to the required quality compared to loosely coupled modules or microservices? What are the advantages and disadvantages of each approach

A **tightly coupled monolithic software** will generally take **longer** to reach the required quality compared to **loosely coupled modules or microservices**, primarily due to **scalability, maintainability, and testing challenges**.

**Comparison: Monolith vs. Modular/Microservices**

|Factor|**Monolithic (Tightly Coupled)**|**Modular/Microservices (Loosely Coupled)**|
|---|---|---|
|**Development Speed**|Faster for **small projects** but slows down as the system grows.|Faster in **large projects** because teams work independently.|
|**Testing & Debugging**|More **complex** due to tight coupling; a small change can require full system testing.|Easier to test and debug since individual modules can be tested separately.|
|**Deployment**|**All-or-nothing deployment**, which increases risk.|**Independent deployments**, allowing for continuous updates.|
|**Scalability**|Hard to scale specific parts without scaling the entire system.|Can scale individual services based on demand.|
|**Fault Tolerance**|A single failure can crash the entire system.|Failures are isolated; one service failing doesn't take down everything.|
|**Code Maintainability**|Harder as the codebase grows; dependencies make modifications risky.|Easier to maintain and upgrade specific parts independently.|
|**Performance**|Often better in simple applications due to fewer network calls.|Can be **slower** if not optimized due to inter-service communication overhead.|
**Advantages & Disadvantages of Each Approach**

**Tightly Coupled Monolith**
**Advantages:**
- Simpler to develop and deploy initially.
- Fewer network-related performance issues.
- Works well for small projects with fewer features.

**Disadvantages:**
- Harder to scale and maintain as the application grows.
- Changes require extensive testing, delaying releases.
- Difficult to adopt new technologies without rewriting large portions of the system.

**Loosely Coupled Modules / Microservices**
**Advantages:**
- Faster development & deployment cycles.
- Easier to scale and maintain.
- Teams can work independently on different services.
- Failures are contained within individual services.

**Disadvantages:**
- Higher complexity in **orchestration & inter-service communication**.
- More **infrastructure & operational overhead** (e.g., API gateways, service discovery).
- Can introduce **latency** due to network calls between services.

**Final Takeaway:**
For **large, evolving projects**, a **modular/microservices** approach is usually **better** as it enables **faster delivery, easier maintenance, and scalability**. However, for **smaller projects**, a **monolithic approach** might be **more efficient** due to **lower complexity and faster initial development**.
