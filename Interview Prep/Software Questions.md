<mark style="background: #D2B3FFA6;">1. consider the effect of modularization on the delivery of software. will it take more or less time for a tightly coupled monolithic software to the required quality compared to loosely coupled modules or microservices? What are the advantages and disadvantages of each approach</mark>

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


2. Define Software feature bloat?
