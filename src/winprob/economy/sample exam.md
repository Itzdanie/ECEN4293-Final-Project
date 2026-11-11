import pypandoc, os, textwrap, json, re
content = r"""
ECEN 4273 SOFTWARE ENGINEERING — EXHAUSTIVE LLM TESTING / QUESTION BANK
Source: ECEN4273_w13_2_Review.pdf (23 pages, Weeks 6–13)
Purpose: Give this document to another LLM as a knowledge test, oral-exam bank, or grading reference.
This is NOT intended to imitate the professor's 16-question / 50-minute exam. It is deliberately much broader.

IMPORTANT SOURCE-FAITHFULNESS RULES
1. The supplied review explicitly identifies the scope as slides W6 through W13 and excludes project-related content.
2. The review names terms and concepts but usually does not define them. The questions below therefore distinguish:
   - SOURCE-EXPLICIT: directly named or visibly represented in the review.
   - STANDARD COURSE-CONTEXT: a conventional software-engineering explanation useful for testing, but not a definition printed in the supplied review.
3. Do not claim that a conventional explanation is the professor's exact wording when the review does not provide it.
4. The review does not expand “CCA” or “CO.” A test-taker should identify that these acronyms are undefined in the supplied source and request/consult lecture notes rather than confidently inventing expansions.
5. The review lists “3 Golden Rules of UI Design” but does not state their wording. Accept the wording taught in the course. A common textbook version is included only as a possible reference, not as a confirmed course quotation.
6. The final nine pages are graphics. They show: (a) repository/blackboard architecture, (b) pipe-and-filter architecture, (c) main-program/subprogram hierarchy, (d) layered architecture with core, utility, application, and UI layers, (e) a login interaction sequence involving Actor, SecurityLogon, AccountDB, and System, (f) an MVC web-app diagram, (g) a pattern organization matrix, (h) integration-order diagrams, and (i) configuration/change-control workflows. Questions about those figures should be grounded in what is visibly drawn.

============================================================
PART A — SCOPE INVENTORY
============================================================

A1. TERMS LISTED IN THE REVIEW

Architecture and design:
- Software Architecture
- Abstraction
- Separation of Concerns
- Functional Independence
- Cohesion
- Coupling
- Design Patterns
- Pattern Language
- Anti-Pattern
- Framework
- Hooks

User experience and interface design:
- UX
- Customer Journey Map
- Touch point
- User Persona

Quality:
- Statistical QA
- Pareto principle
- Six Sigma
- Software reliability

Security:
- Attack Surface
- Common types of attacks

Testing and verification:
- Verification vs. Validation
- Scaffolding
- Stub
- Unit testing
- Integration testing
- Validation testing
- System testing
- Smoke testing
- Regression testing
- Black box testing
- White box testing
- Scenario based testing
- CI/CD
- Thread based testing
- Use-Based testing

Configuration, change, and defect management:
- Configuration management baseline
- Software configuration item
- CCA
- CO
- Misuse and Abuse test cases
- STRIDE threat categories
- Defect Amplification
- Defect Propagation
- Post Mortem Evaluation

A2. CONCEPTS LISTED IN THE REVIEW

- Consequences of bad software architecture design.
- Elements of architectural styles: components, connectors, constraints, models.
- What UX design considers.
- Three Golden Rules of UI Design.
- Error-handling guidelines for UX.
- Aspects of an effective design pattern.
- Kinds of patterns.
- Pattern pitfalls.
- Elements of QA.
- Secure coding practices.
- Why security is important.
- Why use an independent test group?
- Top-down versus bottom-up integration.
- Configuration management elements.
- Depth-first versus breadth-first integration.
- Principle of information hiding.
- Design refactoring.
- Software configuration audit.
- Control structure testing strategies.
- Fundamental guidelines for formal review meetings.
- What reviews are and what reviews are not.

A3. TEST FORMAT CONTEXT (FOR REFERENCE ONLY)
The review says the professor's test is 50 minutes, based on W6–W13, worth 100 points, with 16 questions: 8 multiple choice at 4 points each and 8 short-answer questions worth 8 or 9 points each. One front-and-back cheat sheet is allowed. This bank intentionally exceeds that format.

============================================================
PART B — EXHAUSTIVE QUESTION BANK
============================================================

SECTION 1 — SOFTWARE ARCHITECTURE, ABSTRACTION, AND DESIGN PRINCIPLES

1. Define software architecture. Explain what makes architecture different from the detailed implementation of an individual function or class.
2. Identify the kinds of decisions that belong at the architectural level of a software project.
3. Explain why architecture matters even when the code is technically correct and passes its current tests.
4. Give four possible consequences of poor software architecture design. For each, explain the causal connection between the design flaw and the consequence.
5. Explain how poor architecture can increase the cost or risk of later changes.
6. Explain how architecture affects maintainability, extensibility, testability, reliability, and team coordination.
7. Define abstraction. What information does abstraction intentionally hide, and what information does it preserve?
8. Give a software example in which abstraction allows a user or developer to work without knowing implementation details.
9. Explain the difference between abstraction and information hiding. Are they related? Explain.
10. Define separation of concerns. Explain why unrelated responsibilities should not be mixed in one module or layer.
11. Give an example of a system in which user-interface code, business logic, and data access are separated. Explain the benefit of that separation.
12. Define functional independence.
13. What are cohesion and coupling? Explain how each relates to functional independence.
14. What does high cohesion mean? Give an example of a highly cohesive module.
15. What does low cohesion look like? Give an example of a module with unrelated responsibilities.
16. What does low coupling mean? Explain why a module that depends on many internal details of other modules is highly coupled.
17. What does high coupling look like in a real software design?
18. Why is high cohesion and low coupling generally desirable?
19. Can a design have high cohesion but also high coupling? Explain with an example.
20. Can a design have low cohesion but low coupling? Explain why low coupling alone does not guarantee good design.
21. Explain the principle of information hiding.
22. How does information hiding reduce the effect of changes?
23. Explain how a stable public interface can hide internal implementation decisions.
24. Distinguish encapsulation from information hiding, if the terms are being used in a conventional object-oriented design context.
25. Define design refactoring. Does refactoring necessarily change the externally observable behavior of a program?
26. Explain why refactoring can improve design without adding new features.
27. Describe a safe high-level process for refactoring an existing system.
28. Explain why automated regression tests are useful before and after refactoring.
29. A class reads database records, formats a web page, validates passwords, sends email, and calculates billing totals. Identify design problems using separation of concerns, cohesion, and coupling.
30. A team changes the database schema and must modify dozens of unrelated classes. What architectural/design issue might this indicate? Explain.
31. A module exposes its internal data structures and requires all callers to know how those structures work. Which principle is being violated, and what is the likely consequence?
32. Explain how architecture can make unit testing and integration testing easier or harder.
33. Explain the relationship between architectural constraints and implementation freedom.
34. Give an example of a design decision that should be hidden inside a component rather than exposed to every caller.
35. Explain why architecture should be evaluated in the context of system requirements and constraints rather than by appearance alone.

SECTION 2 — ARCHITECTURAL STYLES AND THE REVIEW'S GRAPHICS

36. Name and explain the four architectural-style elements listed in the review: components, connectors, constraints, and models.
37. What is a component in an architectural description?
38. What is a connector? Give examples of possible communication or coordination mechanisms.
39. What is an architectural constraint? Give examples of rules that restrict component arrangement or interaction.
40. What is an architectural model, and why might multiple models be needed to describe a system?
41. The review shows a central data store labeled “Data store (repository or blackboard)” surrounded by multiple client software boxes with two-way arrows. Identify the architectural style.
42. In a repository/blackboard architecture, what role does the central data store play?
43. How do client components communicate in the repository diagram?
44. Identify two advantages and two disadvantages of a repository architecture.
45. Explain why a central repository can become a bottleneck or a critical failure point.
46. In what kinds of systems might a repository/blackboard architecture be appropriate?
47. The review shows multiple boxes labeled “Filter” connected by arrows labeled “Pipes,” including branching and merging paths. Identify the architectural style.
48. What is the responsibility of a filter in a pipe-and-filter system?
49. What is the responsibility of a pipe?
50. Explain how data moves through the pipe-and-filter diagram.
51. Why can filters often be developed or tested independently?
52. Identify advantages and disadvantages of pipe-and-filter architecture.
53. What problems can occur if adjacent filters disagree about data format or protocol?
54. How does branching or merging affect a pipe-and-filter processing network?
55. Give an example of a real system that could be represented as a pipe-and-filter architecture.
56. The review shows a “Main program” calling several “Controller subprograms,” which in turn call “Application subprograms.” Identify the broad architectural organization.
57. Explain the hierarchical control relationship shown in the main-program/subprogram diagram.
58. What does the diagram suggest about which modules initiate work?
59. What does the shared lower-level application subprogram in the diagram suggest about reuse or dependencies?
60. Give advantages and disadvantages of a call-and-return or main-program/subprogram organization.
61. The review shows concentric layers labeled Core layer, Utility layer, Application layer, and User interface layer. Identify the architecture.
62. Explain the purpose of each layer shown in the review's layered diagram.
63. How should dependencies normally flow in a layered architecture?
64. Why might the core layer be kept independent of the user-interface layer?
65. Give advantages and disadvantages of layered architecture.
66. What is the risk of allowing arbitrary layers to call one another?
67. What is the risk of forcing every interaction through many layers when it is not necessary?
68. Compare repository, pipe-and-filter, call-and-return, and layered architecture. State the organizing idea behind each.
69. For each of the four architectural diagrams, identify the components and connectors.
70. A system has a shared database accessed by many independently developed client applications. Which style from the review is the closest match? Explain.
71. A system processes a stream of input through a sequence of transformations. Which style is the closest match? Explain.
72. A program has a top-level controller that calls subordinate modules in a tree-like hierarchy. Which style is the closest match? Explain.
73. A system groups core services, utilities, application logic, and UI into distinct layers. Which style is the closest match? Explain.
74. Explain why an architectural diagram is a model rather than the implementation itself.
75. What information can be inferred from arrows in an architectural diagram, and what information may remain unspecified?
76. Identify a limitation of trying to understand an architecture from a single diagram.

SECTION 3 — DESIGN PATTERNS, PATTERN LANGUAGES, FRAMEWORKS, AND HOOKS

77. Define a design pattern in software engineering.
78. What kind of recurring problem is a design pattern intended to address?
79. Why is a design pattern not simply a code snippet or a finished software component?
80. What information should a useful pattern description communicate? Consider problem, context, forces, solution, consequences, and implementation considerations.
81. Explain what it means for a pattern to be reusable.
82. Explain why a pattern must be applied in an appropriate context.
83. What makes a design pattern effective? Identify and explain several qualities.
84. Explain how a pattern communicates design knowledge between developers.
85. Define a pattern language.
86. How is a pattern language different from a single design pattern?
87. Explain how patterns can relate to one another or solve problems at different levels of abstraction.
88. The review includes a pattern-organization matrix with categories such as Data/Content, Architecture, Component-level, and User interface, and columns such as Database, Application, Implementation, and Infrastructure. Explain what the matrix is communicating.
89. Why might a pattern catalog organize patterns by both problem area and technical layer?
90. What is an architectural pattern?
91. What is a component-level pattern?
92. What is a user-interface pattern?
93. What is an implementation-level pattern?
94. Explain the difference between an architectural pattern and a lower-level design pattern.
95. Define an anti-pattern.
96. How is an anti-pattern different from a design pattern?
97. Why can an anti-pattern initially appear attractive?
98. Give a hypothetical anti-pattern and explain why it causes problems.
99. Explain at least four pitfalls in using design patterns.
100. Why is “using more patterns” not automatically equivalent to better design?
101. Explain how overengineering can result from applying a pattern without a real need.
102. Explain how misapplying a pattern can increase coupling or complexity.
103. A developer uses a complex pattern to solve a problem that could be handled by a simple function. Evaluate the decision.
104. A team copies a pattern from another system without checking the new system's constraints. What could go wrong?
105. Define a software framework in a conventional software-engineering context.
106. Compare a framework with a library. In particular, discuss who controls the overall flow of execution.
107. Explain how a framework can provide reusable structure while allowing application-specific behavior.
108. What are hooks in a software framework or event-driven design context? Explain that exact course meaning may depend on lecture context.
109. How can hooks allow developers to insert custom behavior at designated points in a framework's execution?
110. What risks arise if hooks are poorly documented, overly numerous, or called in an unexpected order?
111. Explain how a framework, design pattern, and individual hook relate to different levels of software reuse or extension.
112. Identify which claims in this section are directly supported by the review and which require lecture definitions.

SECTION 4 — UX, PERSONAS, CUSTOMER JOURNEY MAPS, TOUCHPOINTS, UI RULES, AND ERROR HANDLING

113. Define user experience (UX).
114. What aspects of a user's interaction should UX design consider?
115. Explain why UX is broader than visual appearance or screen layout.
116. Explain how usability, accessibility, user expectations, context, and user goals can affect UX.
117. Define a user persona.
118. Why should a persona be based on user research rather than an unsupported stereotype?
119. What information might a persona include?
120. Define a customer journey map.
121. What stages or information might a customer journey map represent?
122. Define a touchpoint.
123. Give examples of touchpoints in a software product's customer journey.
124. Explain the relationship among a persona, a customer journey map, and touchpoints.
125. How can a journey map reveal pain points or confusing interactions?
126. How can personas help a development team make design decisions?
127. Why might different personas have different journeys through the same product?
128. The review asks what UX design considers. Explain how a team could investigate user needs before designing an interface.
129. The review lists the “3 Golden Rules of UI Design” but does not provide their wording. State the exact version taught in the course if known. If it is not known, identify that limitation rather than claiming a version is source-confirmed.
130. As a possible conventional reference only, explain the formulation: place the user in control, reduce the user's memory load, and make the interface consistent. Do not assert that this is definitely the lecture's exact wording.
131. Explain what it means to place the user in control of an interface.
132. Explain what it means to reduce the user's memory load.
133. Explain why consistency matters in UI design.
134. How can an interface provide users with clear feedback about the result of an action?
135. Why should an interface make important system states visible?
136. Explain why error messages should be understandable to the intended user.
137. What should a useful error message tell the user?
138. Why should error messages avoid blaming or insulting the user?
139. Explain why preserving user-entered data after an error is important.
140. How can a system prevent errors rather than only reporting them?
141. Explain the difference between preventing an error, detecting an error, and helping the user recover.
142. Give an example of poor error handling and rewrite it as useful guidance.
143. A form rejects a user's submission but clears every field. Evaluate this behavior from a UX perspective.
144. A login page says only “Error 104.” Explain why this is poor error handling and what a better message might communicate without exposing sensitive information.
145. Explain how accessibility needs can affect the design of feedback and error messages.
146. Explain how user testing can reveal problems that developers may overlook.
147. Why is a technically correct interface not necessarily a good user experience?
148. Describe how you would use a journey map to improve a confusing account-registration workflow.

SECTION 5 — QUALITY ASSURANCE, STATISTICAL QA, PARETO, SIX SIGMA, AND RELIABILITY

149. Define quality assurance (QA).
150. Distinguish quality assurance from simply testing the finished product.
151. What are the elements or activities of QA listed or discussed in the course? The review asks for “elements of QA” but does not enumerate them, so use the lecture's list when available.
152. Explain how standards, reviews, testing, measurement, defect tracking, audits, and process improvement can contribute to QA in a conventional process.
153. Define statistical quality assurance.
154. What kinds of software-development data might be collected for statistical QA?
155. How can defect counts, defect types, defect origins, and defect-removal phases be used to identify process problems?
156. Explain why collecting defect data is not enough; the data must be analyzed and acted upon.
157. State the Pareto principle.
158. Explain how the Pareto principle can be applied to software defects.
159. If 20% of defect causes account for 80% of defects, what should a QA team consider doing?
160. Does the Pareto principle mean that exactly 80% of every software problem always comes from exactly 20% of causes? Explain.
161. Explain how Pareto analysis can help prioritize limited testing or process-improvement resources.
162. Define Six Sigma at a high level.
163. What is Six Sigma intended to improve in a process?
164. Explain the relationship between process variation and defects.
165. Why is measurement important to a Six Sigma-style improvement effort?
166. What is software reliability?
167. Explain why reliability must be stated relative to operating conditions, environment, and a time interval.
168. Distinguish reliability from availability, if using conventional definitions.
169. Explain how defect density and failure data may be related to reliability assessment.
170. Why does passing a test suite not prove that software is perfectly reliable?
171. Explain how QA activities can reduce the likelihood of failures after deployment.
172. A product has many minor cosmetic defects and a few severe data-loss defects. Explain why treating all defects as equally important would be a poor QA decision.
173. A team discovers that most production failures originate in one poorly controlled subsystem. Explain how statistical QA and Pareto analysis could guide improvement.
174. Explain why quality must be built into the development process rather than added only at the end.

SECTION 6 — SOFTWARE SECURITY, ATTACK SURFACE, COMMON ATTACKS, SECURE CODING, STRIDE, MISUSE/ABUSE CASES

175. Define software security in the context of a software system.
176. Why is security important to users, organizations, and system operators?
177. Explain the confidentiality, integrity, and availability goals commonly associated with security.
178. Define attack surface.
179. What kinds of interfaces or entry points can contribute to a system's attack surface?
180. Explain why a larger attack surface can create more opportunities for attack.
181. Does reducing the number of exposed interfaces automatically make a system secure? Explain.
182. Identify common types of software attacks that a developer should understand. Use the exact list from lecture if it differs.
183. Explain injection attacks at a high level.
184. Explain how weak authentication can expose a system.
185. Explain how weak authorization differs from weak authentication.
186. Explain how denial-of-service attacks affect availability.
187. Explain how malicious input can exploit poor input validation.
188. Explain why secure coding is needed throughout the software lifecycle.
189. Describe secure input-validation practices.
190. Explain why parameterized database queries are safer than constructing SQL statements by concatenating untrusted strings.
191. Explain the principle of least privilege.
192. Explain why secrets, passwords, and keys should not be hard-coded into source code.
193. Explain why sensitive information should not be exposed in error messages or logs.
194. Explain why dependencies and frameworks should be updated and monitored for known vulnerabilities.
195. Explain the importance of authentication and authorization checks on server-side operations.
196. Explain how secure session management can affect application security.
197. Explain why security testing should include negative and adversarial inputs.
198. Expand and explain each STRIDE category:
   a. Spoofing
   b. Tampering
   c. Repudiation
   d. Information disclosure
   e. Denial of service
   f. Elevation of privilege
199. Give a concrete example of spoofing.
200. Give a concrete example of tampering.
201. Give a concrete example of repudiation.
202. Give a concrete example of information disclosure.
203. Give a concrete example of denial of service.
204. Give a concrete example of elevation of privilege.
205. For each STRIDE category, identify the security property or concern most directly threatened.
206. Distinguish spoofing from elevation of privilege.
207. Distinguish tampering from information disclosure.
208. Distinguish repudiation from spoofing.
209. Explain why one attack scenario may involve more than one STRIDE category.
210. Define a misuse case.
211. Define an abuse case.
212. How do misuse and abuse cases differ from ordinary use cases?
213. Explain how misuse and abuse cases help identify requirements that normal use cases may overlook.
214. Explain how misuse/abuse scenarios can be converted into security test cases.
215. Give a misuse/abuse scenario for a login system.
216. Give a misuse/abuse scenario for an online payment system.
217. Give a misuse/abuse scenario for a system with user roles and permissions.
218. Explain why testing only expected, successful user behavior is insufficient for security.
219. A user changes an account ID in a URL and can view another user's information. Identify the likely security failure and explain a test that could reveal it.
220. A user submits extremely large or malformed requests until the service stops responding. Identify the STRIDE category most directly associated with the outcome.
221. A program logs passwords in plaintext. Explain the security risk.
222. A program trusts a role value supplied by the browser. Explain why this is unsafe.
223. Explain how threat modeling can guide the choice of security tests.
224. Identify which security details are directly stated in the review and which are standard elaborations beyond the review's short topic list.

SECTION 7 — VERIFICATION, VALIDATION, TESTING LEVELS, TEST SUPPORT, AND TESTING METHODS

225. Define verification.
226. Define validation.
227. State the classic distinction between “building the product right” and “building the right product.”
228. Give an example of a verification activity.
229. Give an example of a validation activity.
230. Explain why verification and validation are complementary rather than interchangeable.
231. Define test scaffolding.
232. What kinds of temporary support code or tools may be part of scaffolding?
233. Define a stub.
234. How does a stub help test a component before the real dependency is available?
235. Distinguish a stub from a driver.
236. Explain why test scaffolding should not be confused with production functionality.
237. Define unit testing.
238. What is the target of a unit test?
239. Why should unit tests isolate the unit under test from unrelated dependencies where appropriate?
240. Define integration testing.
241. What does integration testing focus on that unit testing may not reveal?
242. Give examples of interface or interaction defects that integration testing can expose.
243. Define validation testing.
244. What does validation testing attempt to establish?
245. Define system testing.
246. Why is system testing performed on the integrated system?
247. Define smoke testing.
248. Why is smoke testing often performed after a new build or deployment?
249. What does a smoke test generally avoid trying to do?
250. Define regression testing.
251. Why should regression tests be rerun after code changes?
252. Explain the difference between smoke testing and regression testing.
253. Explain the difference between integration testing and system testing.
254. Explain the difference between validation testing and system testing.
255. Explain how unit, integration, validation, system, smoke, and regression testing can complement one another.
256. Define black-box testing.
257. What information does a black-box tester need?
258. What does a black-box tester generally not need to know?
259. Define white-box testing.
260. How does knowledge of source code or internal structure affect white-box test design?
261. Compare black-box and white-box testing.
262. Explain why black-box and white-box tests may find different classes of defects.
263. Define scenario-based testing.
264. How does scenario-based testing relate to realistic user workflows?
265. Can a scenario-based test be black-box? Explain.
266. Can a scenario-based test use white-box knowledge? Explain.
267. Give an example of a scenario-based test for a user registration workflow.
268. Explain why testing only isolated functions may miss problems in a complete user scenario.
269. A function returns the correct result for every unit test, but the application sends the wrong value to it. Which testing level is likely to expose the issue?
270. A new build launches but a critical login operation immediately crashes. What kind of quick test should detect this before extensive testing?
271. A code change causes an older feature to stop working. What kind of testing is intended to detect this?
272. A system meets its internal design but users cannot complete the task they need. Is this primarily a verification or validation concern? Explain.
273. Explain why test cases should include expected results and clear pass/fail criteria.
274. Explain why testing cannot prove the absence of every possible defect.
275. Describe how test results should be recorded so that failures can be reproduced and investigated.

SECTION 8 — CONTROL-STRUCTURE TESTING STRATEGIES

276. What is control-structure testing?
277. Explain statement coverage.
278. Explain branch or decision coverage.
279. Explain condition coverage.
280. Explain path coverage.
281. How does statement coverage differ from branch coverage?
282. How does branch coverage differ from condition coverage?
283. Why can 100% statement coverage still leave important logical defects undiscovered?
284. Why is complete path coverage often impractical for programs with loops or many branches?
285. Explain how control-flow graphs can help design white-box tests.
286. Given a conditional with true and false outcomes, what tests are needed for branch coverage?
287. Explain how boundary conditions can be incorporated into control-structure testing.
288. A test suite executes every line but never exercises the false outcome of a decision. What coverage measure is incomplete?
289. A compound decision contains two Boolean conditions. Explain why testing only the overall true and false outcomes may fail to test each condition independently.
290. Explain why control-structure testing is a white-box strategy.
291. Identify the exact control-structure testing strategies emphasized in lecture if the instructor used a different named list; the supplied review only says “Control Structure testing strategies.”

SECTION 9 — CI/CD AND INTEGRATION TESTING STRATEGIES

292. Expand CI/CD.
293. Explain continuous integration.
294. Explain continuous delivery.
295. Explain continuous deployment.
296. Distinguish continuous delivery from continuous deployment.
297. Describe a conventional CI pipeline from a developer's code change through automated build and tests.
298. Explain how CI can identify integration problems earlier.
299. Explain how automated testing supports CI/CD.
300. Does CI/CD eliminate the need for code review, manual testing, or security review? Explain.
301. What can go wrong when a team integrates code infrequently?
302. Explain how CI/CD relates to regression testing.
303. Define top-down integration testing.
304. In top-down integration, why might stubs be needed?
305. What are the advantages of top-down integration?
306. What are the disadvantages of top-down integration?
307. Define bottom-up integration testing.
308. In bottom-up integration, why might drivers be needed?
309. What are the advantages of bottom-up integration?
310. What are the disadvantages of bottom-up integration?
311. Compare top-down and bottom-up integration.
312. Define depth-first integration.
313. Define breadth-first integration.
314. Explain the order in which modules are integrated in depth-first integration.
315. Explain the order in which modules are integrated in breadth-first integration.
316. Compare depth-first and breadth-first integration.
317. Can depth-first/breadth-first be considered the same distinction as top-down/bottom-up? Explain why they describe different dimensions of integration order.
318. The review includes an integration diagram with a hierarchy of modules and dashed or solid connections. Explain how a diagram can represent different integration orders.
319. A team wants to test one complete branch of a hierarchy early. Which integration strategy may be appropriate?
320. A team wants to integrate all modules at a particular level before moving down. Which strategy may be appropriate?
321. Explain how stubs and drivers support incremental integration.
322. Explain why integration order affects when defects become visible.
323. What is the role of an independent test group?
324. Why might developers overlook defects in their own code?
325. What benefits can an independent testing group provide?
326. What communication problems can occur between developers and independent testers?
327. Why should an independent test group not be treated as a substitute for developer testing or QA throughout development?
328. Describe how an independent test group can maintain objectivity while still collaborating with developers.

SECTION 10 — THREAD-BASED AND USE-BASED TESTING

329. The review lists thread-based testing. Define it using the course's terminology.
330. Explain what a “thread” may represent in a software system or test strategy.
331. How might a test thread follow a sequence of connected components through the architecture?
332. What kinds of end-to-end defects might thread-based testing reveal?
333. The review lists use-based testing. Define it using the course's terminology.
334. Explain how use-based testing may prioritize functionality or components according to how the system is used.
335. Compare thread-based testing and use-based testing.
336. Give a hypothetical example of a thread-based test.
337. Give a hypothetical example of a use-based test.
338. Explain how thread-based and use-based strategies can complement one another.
339. The supplied review names these strategies but does not define them. If the course's exact meanings differ from conventional interpretations, identify the need for lecture notes rather than inventing an exact definition.

SECTION 11 — CONFIGURATION MANAGEMENT, BASELINES, SCIs, CHANGE CONTROL, AND AUDITS

340. Define software configuration management.
341. Why is configuration management important in a software project with multiple developers and multiple versions?
342. Define a software configuration item (SCI).
343. Give examples of artifacts that may be treated as SCIs.
344. Why should configuration items be identified and controlled?
345. Define a configuration management baseline.
346. Why is a baseline an important reference point?
347. Explain how a baseline differs from an informal working copy.
348. What kinds of changes should be controlled after a baseline is established?
349. Identify the major configuration-management elements named in the course. The review asks for these elements but does not enumerate them.
350. In a conventional configuration-management process, explain configuration identification.
351. Explain configuration change control.
352. Explain configuration status accounting.
353. Explain configuration audits.
354. Define a software configuration audit.
355. What questions should an audit answer about a software product and its configuration records?
356. Explain the difference between a functional configuration audit and a physical configuration audit if those terms are part of the course material.
357. Why should a software configuration audit verify that approved changes are actually present?
358. Why should it verify that unapproved changes are absent?
359. Explain how version control supports configuration management.
360. Explain how change requests should be evaluated, approved, tracked, and incorporated.
361. The review's configuration workflow diagram begins with a developer evaluating a change request. Describe the visible flow: request accepted or denied; if denied, user is informed; if accepted, configuration items are assigned/checked, the change is made and reviewed/audited, changed items are checked, a testing baseline is established, QA/testing occurs, changes are promoted for the next release, a new version is rebuilt and audited, and the new version is distributed.
362. Why does the configuration workflow include both checking changed items and establishing a testing baseline?
363. Why are changes reviewed or audited before release?
364. Explain why rebuilding a new version from controlled configuration items is important.
365. What problems can occur if a team distributes a build that cannot be reproduced from its recorded configuration?
366. The review shows a second change-control workflow with four classes of change. Describe what the diagram indicates about different change classes and levels of review/approval.
367. Why might a minor change require less evaluation than a change affecting many related objects or stakeholders?
368. Explain why change classification should consider impact and dependencies.
369. What does it mean to maintain traceability between a change request and the configuration items changed?
370. Explain the relationship between configuration management and regression testing.
371. Explain the relationship between configuration management and release management.
372. The review includes the acronyms CCA and CO but does not expand them. State that the source does not define them. Do not invent expansions; request the lecture notes or instructor's terminology.
373. If an LLM confidently expands CCA or CO without evidence from the review, explain why that answer is not source-grounded.

SECTION 12 — DEFECT AMPLIFICATION, DEFECT PROPAGATION, AND POST-MORTEM EVALUATION

374. Define a software defect.
375. Explain what defect amplification means in the context used by the course.
376. Explain what defect propagation means in the context used by the course.
377. Compare defect amplification and defect propagation. Make the distinction explicit.
378. Give an example in which a small requirements defect causes a larger design or implementation problem later.
379. Give an example in which a defect in one artifact is carried into a downstream artifact.
380. Explain why defects discovered late can be more expensive to correct.
381. How can reviews and early testing reduce defect propagation?
382. How can unclear requirements amplify defects across a project?
383. Explain how defect tracking can help identify the stage where defects were introduced and the stage where they were detected.
384. Define post-mortem evaluation.
385. What is the purpose of a post-mortem after a project, release, incident, or major defect?
386. What questions should a post-mortem attempt to answer?
387. Why should a post-mortem focus on process and evidence rather than simply assigning blame?
388. Explain how lessons from a post-mortem should be converted into concrete process improvements.
389. What is the difference between identifying a root cause and merely describing the symptom?
390. Explain why a post-mortem should document both successful practices and failures.
391. Give an example of a useful action item resulting from a post-mortem.
392. Explain how post-mortem findings can feed statistical QA and continuous improvement.

SECTION 13 — FORMAL REVIEWS AND REVIEW MEETINGS

393. Define a formal software review.
394. What is the main purpose of a review?
395. Explain why reviews are intended to find defects or improve a work product rather than judge the author.
396. Identify fundamental guidelines for a formal review meeting.
397. Why should the review have a defined scope and objective?
398. Why should participants receive the material in advance?
399. Why should participants prepare individually before the meeting?
400. Why should roles and responsibilities be clear?
401. Why should a review meeting follow an agenda?
402. Why should the review focus on identifying and recording issues rather than solving every issue during the meeting?
403. Why should findings be documented?
404. Why is follow-up necessary after a review?
405. Explain why the author should not be placed in a defensive position.
406. Explain why reviewers should avoid turning a review into a debate about personal preferences.
407. What is a review not?
408. Explain why a review is not simply a casual proofreading session.
409. Explain why a review is not a substitute for testing.
410. Explain why a review is not a guarantee that a product contains no defects.
411. Explain how reviews differ from audits.
412. Explain how reviews differ from testing.
413. Describe how a review can detect defects in requirements, design, code, test plans, or documentation.
414. A meeting spends most of its time debating possible solutions and records no defects or action items. Evaluate the meeting against formal-review guidelines.
415. A reviewer attacks the developer personally after finding a defect. Explain why this violates the purpose of a review.
416. Explain how review findings can become inputs to defect tracking, QA measurement, and process improvement.

SECTION 14 — GRAPHIC INTERPRETATION QUESTIONS

417. Repository diagram: identify the central cylinder and surrounding client boxes. Explain the arrows and the relationship between clients and the data store.
418. Pipe-and-filter diagram: identify filters and pipes. Trace one possible path from input to output and explain where branching or merging occurs.
419. Main-program diagram: identify the root, controller subprograms, and application subprograms. Explain the hierarchy and the shared lower-level subprogram.
420. Layered diagram: identify the four named layers from innermost to outermost: Core, Utility, Application, and User Interface. Explain the conceptual separation.
421. Login interaction diagram: identify the actor and the three objects labeled SecurityLogon, AccountDB, and System. Trace the numbered calls:
   - Actor requests logon and sends Logon(name, pass).
   - SecurityLogon calls displayLogon(), displayMessage(), and display().
   - SecurityLogon calls isValid(): boolean on AccountDB.
   - AccountDB calls userType(name): string on System.
   Explain the order and purpose of the interactions without claiming details not shown.
422. What does the login diagram suggest about the responsibilities of SecurityLogon, AccountDB, and System?
423. What does the return type “boolean” on isValid() tell you? What does the return type “string” on userType(name) tell you?
424. MVC diagram: identify Model, View, and Controller.
425. In the MVC diagram, what responsibilities are assigned to the Controller?
426. In the MVC diagram, what responsibilities are assigned to the Model?
427. In the MVC diagram, what responsibilities are assigned to the View?
428. Explain the relationship between the browser/client and the server in the MVC diagram.
429. Explain the diagram's data flow: browser user request/data to Controller; controller behavior request/state change to Model; view selection to View; model data to View; view update request to Model; Model and external data interaction; HTML data back to browser.
430. Why does the MVC separation help organize a web application?
431. Which MVC component is responsible for selecting model behavior and selecting the view response in the diagram?
432. Which MVC component encapsulates functionality, content objects, and WebApp state in the diagram?
433. Which MVC component prepares data from the model and presents the view selected by the controller?
434. Pattern matrix: identify the row groupings Data/Content, Architecture, Component-level, and User interface.
435. Identify the column groupings Database, Application, Implementation, and Infrastructure.
436. What does the matrix suggest about organizing patterns by recurring problem statement and technical area?
437. Integration-order graphic: explain how the left hierarchy represents module relationships and how solid/dashed lines may indicate selected versus not-yet-integrated modules. Do not assume a legend not shown in the source.
438. Compare the two integration-order illustrations on the page: one organized around a module hierarchy and another showing clusters. Explain what the diagrams are intended to help reason about.
439. Configuration workflow graphic: describe the left-hand flow from evaluating a change request through release/distribution.
440. Configuration workflow graphic: describe the right-hand flow for classifying changes, assessing impacts, obtaining review/approval, changing objects, checking them, and publishing the WebApp.
441. What is the relationship between configuration item review, change impact assessment, testing, and release in the workflow?
442. Explain what information the graphics provide that a term list alone would not provide.

============================================================
PART C — MULTIPLE-CHOICE QUESTION BANK
============================================================

For each question, select the single best answer.

MC1. Which is the best description of software architecture?
A. The syntax of every source-code statement.
B. The high-level organization of system components and their relationships.
C. The final list of defects found during testing.
D. The visual theme of the user interface.
Correct: B.

MC2. Which combination best supports functional independence?
A. Low cohesion and high coupling.
B. High cohesion and low coupling.
C. High cohesion and high coupling.
D. Low cohesion and low coupling.
Correct: B.

MC3. Which item is one of the four architectural-style elements listed in the review?
A. Stakeholder salary.
B. Connector.
C. Test score.
D. Sprint velocity.
Correct: B.

MC4. In the repository diagram, where do client software components interact?
A. Only with one another directly.
B. Through the shared data store.
C. Through a single user persona.
D. Through a pipe between every pair of clients.
Correct: B.

MC5. In pipe-and-filter architecture, what carries data between filters?
A. Personas.
B. Pipes.
C. Baselines.
D. Stubs.
Correct: B.

MC6. Which architecture is shown by a main program calling controller subprograms that call application subprograms?
A. Repository.
B. Pipe-and-filter.
C. Hierarchical call-and-return.
D. Peer-to-peer mesh.
Correct: C.

MC7. Which set matches the four layers shown in the review?
A. Core, Utility, Application, User interface.
B. Database, Testing, Security, Release.
C. Input, Output, Storage, Network.
D. User, Admin, Developer, Tester.
Correct: A.

MC8. Which statement best describes abstraction?
A. Exposing every internal detail.
B. Hiding irrelevant details while emphasizing essential characteristics.
C. Removing all interfaces.
D. Replacing tests with documentation.
Correct: B.

MC9. What is the main purpose of information hiding?
A. Make all internal data globally accessible.
B. Isolate internal design decisions behind interfaces.
C. Prevent developers from testing modules.
D. Increase the number of dependencies.
Correct: B.

MC10. A design pattern is best understood as:
A. A reusable solution description for a recurring design problem in context.
B. A required source-code file extension.
C. A complete application that needs no adaptation.
D. A testing metric.
Correct: A.

MC11. A pattern language is:
A. A programming language used to implement patterns.
B. A related system of patterns and their relationships.
C. A list of compiler errors.
D. A UI translation file.
Correct: B.

MC12. An anti-pattern is:
A. A pattern that has never been documented.
B. A recurring approach that appears useful but tends to produce harmful consequences.
C. Any design pattern written in another language.
D. A formal review checklist.
Correct: B.

MC13. Which is a pattern-use pitfall?
A. Evaluating context before choosing a pattern.
B. Considering consequences.
C. Applying a complex pattern where a simple solution is sufficient.
D. Documenting the problem being solved.
Correct: C.

MC14. Which statement best captures UX?
A. UX is only the color and typography of a screen.
B. UX considers the user's overall interaction, goals, context, and experience.
C. UX is only backend performance.
D. UX is the same as software architecture.
Correct: B.

MC15. A persona is:
A. A research-based representation of a target user.
B. A software configuration item.
C. A test driver.
D. A database connector.
Correct: A.

MC16. A customer journey map primarily represents:
A. The source-code call graph.
B. Stages and interactions in a user's experience.
C. The release version history.
D. The list of compiler warnings.
Correct: B.

MC17. A touchpoint is:
A. A specific interaction between a user and a product or organization.
B. A module dependency.
C. A configuration baseline.
D. A type of software attack.
Correct: A.

MC18. Which is a useful error-handling practice?
A. Clear all user input after any error.
B. Display only an unexplained numeric code.
C. Explain the problem and how to recover while preserving user input where possible.
D. Blame the user.
Correct: C.

MC19. Statistical QA is primarily concerned with:
A. Collecting and analyzing quality-related data.
B. Choosing UI colors.
C. Eliminating the need for testing.
D. Selecting a software architecture by popularity.
Correct: A.

MC20. The Pareto principle is commonly summarized as:
A. Every cause contributes equally.
B. A relatively small number of causes may account for a large share of effects.
C. Every defect costs the same amount.
D. Software must have exactly six layers.
Correct: B.

MC21. Six Sigma is broadly associated with:
A. Increasing process variation.
B. Reducing defects and process variation through measurement and improvement.
C. Replacing QA with user surveys.
D. A six-person review group.
Correct: B.

MC22. Software reliability concerns:
A. The probability of failure-free operation under specified conditions for a specified time.
B. Whether a UI looks modern.
C. The number of developers on a project.
D. Whether every source file has comments.
Correct: A.

MC23. Attack surface refers to:
A. The visual area of a screen.
B. The exposed entry points and interfaces through which a system may be attacked.
C. The number of test cases in a suite.
D. The physical size of a server.
Correct: B.

MC24. Which STRIDE category involves impersonation?
A. Spoofing.
B. Tampering.
C. Repudiation.
D. Denial of service.
Correct: A.

MC25. Which STRIDE category involves unauthorized modification?
A. Information disclosure.
B. Tampering.
C. Spoofing.
D. Elevation of privilege.
Correct: B.

MC26. Which STRIDE category involves denying an action when reliable evidence is insufficient?
A. Repudiation.
B. Denial of service.
C. Spoofing.
D. Tampering.
Correct: A.

MC27. Which STRIDE category involves exposing confidential information?
A. Elevation of privilege.
B. Information disclosure.
C. Repudiation.
D. Spoofing.
Correct: B.

MC28. Which STRIDE category concerns making a service unavailable?
A. Denial of service.
B. Tampering.
C. Spoofing.
D. Repudiation.
Correct: A.

MC29. Which STRIDE category concerns gaining permissions beyond those authorized?
A. Information disclosure.
B. Elevation of privilege.
C. Repudiation.
D. Spoofing.
Correct: B.

MC30. Misuse and abuse test cases are useful because they:
A. Test only the ideal successful workflow.
B. Exercise hostile, invalid, or unintended behavior.
C. Replace all unit tests.
D. Prove a system is completely secure.
Correct: B.

MC31. Verification asks whether:
A. The product meets actual user needs.
B. The product is being built according to specified requirements.
C. The customer likes the color.
D. The project has finished.
Correct: B.

MC32. Validation asks whether:
A. The correct product is being built for its intended use and needs.
B. Every line has been executed.
C. The compiler can parse the code.
D. The repository has a baseline.
Correct: A.

MC33. A stub is:
A. A simplified substitute for a component called by the unit under test.
B. A complete production database.
C. A formal review meeting.
D. A security threat.
Correct: A.

MC34. Unit testing focuses on:
A. An individual unit or component.
B. The entire organization.
C. Only deployment infrastructure.
D. Customer journey maps.
Correct: A.

MC35. Integration testing focuses on:
A. Whether components work together correctly.
B. Only the appearance of a screen.
C. Whether a project has a post-mortem.
D. Only one isolated function.
Correct: A.

MC36. System testing evaluates:
A. The complete integrated system.
B. Only one stub.
C. Only a single branch in a method.
D. Only user personas.
Correct: A.

MC37. Smoke testing is generally:
A. A quick check that a build's critical functions work.
B. An exhaustive test of every possible path.
C. A test of physical smoke detectors.
D. A review of source-code formatting.
Correct: A.

MC38. Regression testing checks:
A. Whether changes have broken previously working functionality.
B. Whether the user persona is realistic.
C. Whether the original requirements were written in English.
D. Whether a test driver exists.
Correct: A.

MC39. Black-box testing is based primarily on:
A. External behavior and specifications, without requiring internal implementation knowledge.
B. Complete knowledge of every source-code path.
C. Only compiler internals.
D. Only database schema.
Correct: A.

MC40. White-box testing uses:
A. Knowledge of internal structure and control flow.
B. Only user interviews.
C. Only customer journey maps.
D. No expected results.
Correct: A.

MC41. Scenario-based testing focuses on:
A. Realistic workflows or situations of use.
B. Only isolated arithmetic statements.
C. Only configuration audits.
D. Only code style.
Correct: A.

MC42. CI/CD is associated with:
A. Automating integration, testing, and delivery/deployment workflows.
B. Replacing software architecture.
C. Eliminating the need for version control.
D. A user-interface pattern.
Correct: A.

MC43. Top-down integration usually begins with:
A. High-level modules and uses stubs for lower-level modules not yet integrated.
B. The lowest-level modules and uses no test support.
C. The user manual.
D. The release audit.
Correct: A.

MC44. Bottom-up integration usually begins with:
A. Lower-level modules and may use drivers to exercise them.
B. The top-level user interface only.
C. The final production release.
D. The post-mortem.
Correct: A.

MC45. Depth-first integration generally:
A. Completes one branch or path down the hierarchy before moving to another.
B. Integrates every module at the same level before descending.
C. Tests only the UI.
D. Avoids module hierarchies.
Correct: A.

MC46. Breadth-first integration generally:
A. Integrates modules across a level before moving deeper.
B. Completes one branch entirely before any other branch.
C. Tests only leaf nodes.
D. Is identical to regression testing.
Correct: A.

MC47. A configuration item is:
A. An artifact placed under configuration control.
B. Any informal conversation.
C. A user persona.
D. A UI error message.
Correct: A.

MC48. A configuration baseline is:
A. An approved reference configuration against which changes are controlled.
B. A temporary untracked file.
C. A list of test failures only.
D. A design pattern.
Correct: A.

MC49. A software configuration audit checks:
A. Whether the product/configuration and its records conform to requirements and approved changes.
B. Whether the UI is fashionable.
C. Whether every developer agrees with every design decision.
D. Whether all defects have been hidden.
Correct: A.

MC50. A post-mortem evaluation is intended to:
A. Learn from what happened and identify improvements.
B. Assign personal blame as its only purpose.
C. Replace all future testing.
D. Change source code without review.
Correct: A.

MC51. A formal review should primarily:
A. Identify and record issues in a work product.
B. Judge the author's personality.
C. Guarantee that no defects remain.
D. Replace system testing.
Correct: A.

MC52. Which statement best describes a review?
A. A structured examination of a work product.
B. A guarantee of perfect software.
C. A substitute for all testing.
D. A personal performance evaluation of the author.
Correct: A.

MC53. Which testing strategy uses internal control-flow information?
A. White-box.
B. Black-box.
C. User-persona mapping.
D. Configuration audit.
Correct: A.

MC54. Which activity is most likely to detect a defect caused by two modules disagreeing about an interface?
A. Integration testing.
B. Persona creation.
C. Post-mortem only.
D. UI color review.
Correct: A.

MC55. Which activity most directly checks whether a change was properly approved and recorded?
A. Configuration management/change control.
B. Scenario-based UX testing.
C. Pipe-and-filter processing.
D. Unit testing only.
Correct: A.

MC56. Which is the best reason for an independent test group?
A. It can provide a more objective perspective and identify defects overlooked by developers.
B. It makes developers unnecessary.
C. It removes the need for requirements.
D. It guarantees no defects.
Correct: A.

============================================================
PART D — SCENARIO-BASED APPLICATION TESTS
============================================================

For each scenario, require the respondent to identify relevant concepts, explain why they apply, and propose an appropriate response or test.

S1. A single “Manager” class handles login, database queries, billing, email, UI formatting, and report generation. Analyze the design using cohesion, coupling, separation of concerns, and information hiding. Propose a more maintainable organization.

S2. A software system has six client programs that all read and write the same central data store. Identify the architecture shown in the review that most closely matches this design. Explain benefits, risks, and possible failure points.

S3. A data-processing system receives a file, validates it, transforms records, filters invalid entries, aggregates results, and exports a report. Identify the most appropriate architectural style from the review and explain how the stages communicate.

S4. A program has a top-level controller that calls three subsystem controllers. Each controller calls application modules. Identify the architectural organization and explain how a change to a shared lower-level module might affect multiple branches.

S5. A web application is organized into Core, Utility, Application, and User Interface layers. A developer puts database-specific logic directly in the UI. Explain why this may violate the intended separation and how the design could be improved.

S6. A team chooses a complex design pattern because it is popular, even though the problem is simple. Identify the pattern pitfall and explain how to decide whether a pattern is appropriate.

S7. A product team has designed a workflow without interviewing users. The interface looks polished but users cannot complete the main task. Explain the UX failure and how personas, journey maps, and touchpoints could help.

S8. A form reports “Invalid request” and clears all fields. Identify error-handling problems and propose a better approach.

S9. QA data show that most defects come from a small number of recurring causes. Explain how statistical QA and the Pareto principle can guide a process-improvement effort.

S10. A team measures defects but never reviews trends or changes its process. Explain why this is not sufficient statistical QA.

S11. A system's service becomes unavailable when it receives a flood of requests. Identify the STRIDE category most closely associated with the result and explain what security tests could reveal the weakness.

S12. A user changes a record identifier in a URL and retrieves another user's private record. Explain the likely authorization failure, identify relevant security concerns, and create a misuse/abuse test case.

S13. An attacker submits a modified request that changes an account balance. Identify the relevant STRIDE category and propose a test that could reveal the vulnerability.

S14. A user claims they never approved a transaction, and the system has no reliable audit record. Identify the relevant STRIDE category and explain the importance of logging and evidence.

S15. A test suite confirms each function works individually, but the complete workflow fails because one component sends a different data format than another expects. Identify the testing level most directly intended to expose this issue.

S16. A team integrates all modules at once near the end of development. Explain the risks and how CI and incremental integration could reduce them.

S17. A top-level module is being tested before its lower-level dependencies are complete. Explain why stubs may be used and what they need to simulate.

S18. A low-level database module is ready, but its calling application module is not. Explain how a driver could help test the database module in a bottom-up strategy.

S19. A new build starts successfully but its main login function crashes. Explain why a smoke test should catch this before a larger test cycle.

S20. A developer fixes a bug in payment processing, but an older checkout workflow stops working. Identify the test type that should detect the unintended side effect.

S21. A system satisfies every written technical specification but does not solve the user's actual problem. Explain the difference between verification and validation and identify the failure.

S22. A project changes a source file but does not record the change or update the baseline. Explain configuration-management risks.

S23. A release is built from files on one developer's machine and cannot be recreated. Explain the role of SCIs, baselines, version control, and configuration audits.

S24. A change request affects several related components and external stakeholders. Explain why impact assessment and a higher level of review may be appropriate.

S25. A formal review meeting becomes a debate about the developer's competence. Explain what the review is supposed to accomplish and how the meeting should be redirected.

S26. A project ends after a serious production incident. Describe a post-mortem process that identifies root causes and creates useful improvement actions without focusing on blame.

S27. A test suite executes every statement in a function but never exercises the false branch of a decision. Explain the coverage gap and propose a test.

S28. A conditional has two Boolean conditions. The test suite exercises the overall decision as true and false but does not independently vary each condition. Explain why condition coverage may still be incomplete.

S29. A team uses black-box tests based on requirements but never examines internal branches. Explain what white-box tests could add.

S30. A team has thorough unit tests but no end-to-end user workflows. Explain what scenario-based testing could reveal.

============================================================
PART E — SHORT-ANSWER / EXPLANATION PROMPTS
============================================================

1. Explain the difference between architecture and detailed design.
2. Explain abstraction using a concrete software example.
3. Explain separation of concerns and information hiding. How do they reinforce one another?
4. Explain why high cohesion and low coupling support functional independence.
5. Describe four consequences of poor architecture.
6. Explain all four elements of architectural styles.
7. Compare repository and pipe-and-filter architecture.
8. Compare call-and-return and layered architecture.
9. Explain how the MVC diagram separates responsibilities.
10. Explain what makes a design pattern effective.
11. Explain how a pattern language differs from a pattern catalog.
12. Explain three pattern pitfalls.
13. Explain the difference between a framework and a library.
14. Explain what hooks do in a framework, while acknowledging that exact course usage should be checked.
15. Explain UX and how it differs from UI appearance.
16. Explain how persona, journey map, and touchpoint relate.
17. State the three UI golden rules as taught in the course; identify uncertainty if exact wording is unavailable.
18. Explain five guidelines for useful error handling.
19. Explain statistical QA and how it uses defect data.
20. Explain the Pareto principle in defect prioritization.
21. Explain the purpose of Six Sigma.
22. Define software reliability.
23. Explain attack surface and why it matters.
24. Explain five secure-coding practices.
25. Explain all six STRIDE categories with examples.
26. Explain how misuse and abuse test cases improve security testing.
27. Compare verification and validation.
28. Explain scaffolding and stubs.
29. Compare unit, integration, validation, and system testing.
30. Compare smoke and regression testing.
31. Compare black-box, white-box, and scenario-based testing.
32. Explain statement, branch, condition, and path coverage.
33. Explain CI/CD and its relationship to automated testing.
34. Compare top-down and bottom-up integration.
35. Compare depth-first and breadth-first integration.
36. Explain why independent testing can be valuable.
37. Explain thread-based and use-based testing using the instructor's definitions.
38. Define SCI and configuration baseline.
39. Explain the elements of configuration management.
40. Explain the purpose of software configuration audits.
41. Explain the visible configuration workflow in the review.
42. Explain CCA and CO only if their expansions are available from lecture; otherwise state that the supplied review does not define them.
43. Compare defect amplification and defect propagation.
44. Explain post-mortem evaluation.
45. Explain the fundamental guidelines for formal reviews.
46. Explain what reviews are and what they are not.
47. Explain how reviews, QA, testing, configuration management, and post-mortems fit together in a quality process.

============================================================
PART F — ANSWER EXPECTATIONS / GRADING RUBRIC
============================================================

Use this rubric when another LLM evaluates responses.

GENERAL SCORING
- Full credit: accurate definition, correct distinction from nearby concepts, and a relevant example or application when requested.
- Strong partial credit: core idea is correct but explanation is incomplete or lacks an example.
- Limited credit: uses related vocabulary but confuses the concepts or omits the central distinction.
- No credit: incorrect definition, circular explanation, or unsupported claim presented as certain.
- Source-faithfulness penalty: deduct when the respondent attributes a definition to the supplied review that is not actually present in it.
- Do not penalize a respondent for saying that CCA, CO, or the exact golden-rule wording is absent from the source. That is the source-faithful response.

CORE DISTINCTIONS THAT MUST BE PRESERVED
1. Architecture vs. detailed implementation: system-level organization versus internal code-level details.
2. Abstraction vs. information hiding: simplifying essential view versus concealing internal decisions behind interfaces.
3. Cohesion vs. coupling: relatedness within a module versus dependency between modules.
4. High cohesion + low coupling: generally supports functional independence.
5. Component vs. connector vs. constraint vs. model: architectural unit vs. interaction mechanism vs. restriction vs. representation.
6. Repository vs. pipe-and-filter vs. call-and-return vs. layered: shared store vs. transformation pipeline vs. hierarchical invocation vs. organized layers.
7. Design pattern vs. pattern language vs. anti-pattern: recurring solution description vs. related system of patterns vs. recurring harmful approach.
8. Framework vs. library: framework often controls overall flow and calls application extension points; a library is generally called by application code. Treat this as standard context unless lecture wording is supplied.
9. Persona vs. journey map vs. touchpoint: user representation vs. experience over stages vs. individual interaction.
10. UX vs. UI: overall user experience versus interface through which interaction occurs.
11. Verification vs. validation: conformance to specification versus suitability for intended user needs.
12. Stub vs. driver: substitute called component versus test code that calls/exercises the component under test.
13. Unit vs. integration vs. system vs. validation: isolated unit vs. component interactions vs. complete integrated system vs. satisfaction of intended requirements/needs. Course wording may distinguish validation testing more specifically.
14. Smoke vs. regression: quick critical-build check vs. checking for unintended effects of changes.
15. Black-box vs. white-box: external behavior/specification vs. internal structure/control flow.
16. Top-down vs. bottom-up: integration begins at high-level modules vs. lower-level modules.
17. Depth-first vs. breadth-first: one hierarchy branch at a time vs. integration across a level. Do not collapse this distinction into top-down/bottom-up.
18. SCI vs. baseline: controlled artifact vs. approved reference configuration.
19. Defect amplification vs. propagation: increasing downstream impact/cost/number versus a defect being carried into or affecting downstream artifacts. Use course-specific definitions if available.
20. Review vs. testing vs. audit vs. post-mortem: examination of work product vs. execution-based evaluation vs. configuration/conformance check vs. retrospective learning.

STRIDE REFERENCE
- Spoofing: impersonating another entity.
- Tampering: unauthorized modification of data or code.
- Repudiation: denying an action when reliable evidence/auditing is inadequate.
- Information disclosure: exposing information to an unauthorized party.
- Denial of service: impairing availability.
- Elevation of privilege: obtaining permissions beyond those authorized.

SECURE-CODING EXAMPLES THAT ARE ACCEPTABLE IN GENERAL
- Validate untrusted input.
- Use parameterized queries rather than unsafe string concatenation.
- Enforce authentication and authorization server-side.
- Apply least privilege.
- Protect passwords, secrets, and cryptographic keys.
- Avoid exposing sensitive data in errors or logs.
- Keep dependencies updated and monitor known vulnerabilities.
- Use secure session handling.
These are standard examples for elaboration; the review itself only names secure coding practices and security topics.

QA REFERENCE
- Statistical QA requires collecting and analyzing quality/process data, not just counting defects.
- Pareto analysis helps identify a relatively small set of high-impact causes.
- Six Sigma is associated with reducing process variation and defects through measurement and improvement.
- Reliability is conditional on operating environment and time period.
- The review does not enumerate the exact “elements of QA”; accept the course's actual list when supplied.

TESTING REFERENCE
- Unit: individual unit.
- Integration: interfaces/interactions among units.
- Validation: whether requirements/intended use are satisfied; use the instructor's exact level/definition if more specific.
- System: complete integrated system.
- Smoke: quick check of critical functionality/build viability.
- Regression: detect unintended breakage after change.
- Black-box: external behavior.
- White-box: internal logic/control structure.
- Scenario-based: realistic workflow/use situation.
- Statement coverage: execute statements.
- Branch coverage: exercise decision outcomes.
- Condition coverage: exercise Boolean subconditions.
- Path coverage: execute control-flow paths; complete path coverage can be impractical with loops and combinations.

CONFIGURATION MANAGEMENT REFERENCE
Conventional elements often include:
- Configuration identification.
- Change control.
- Configuration status accounting.
- Configuration audits.
The review asks for “configuration management elements” but does not print the exact list. Treat this as standard context unless the lecture notes supply the course list.
A configuration workflow should include request evaluation, acceptance/denial, impact assessment, assignment/identification of affected items, controlled modification, review/audit, testing, baseline establishment, version rebuild, and release/distribution as shown in the graphics.

FORMAL REVIEW REFERENCE
A strong answer should mention:
- Defined purpose and scope.
- Advance distribution and individual preparation.
- Clear roles and agenda.
- Focus on the work product, not the author.
- Identify and record defects/issues.
- Avoid spending the meeting trying to solve every issue.
- Document decisions and action items.
- Follow up to confirm resolution.
A review is not a personal attack, a guarantee of perfection, or a replacement for testing.

SOURCE LIMITATIONS / ITEMS REQUIRING LECTURE NOTES
- “CCA”: appears in the terms list; not expanded.
- “CO”: appears in the terms list; not expanded.
- “3 Golden Rules of UI Design”: named, wording absent.
- “Elements of QA”: asked about, but not enumerated.
- “Configuration management elements”: asked about, but not enumerated.
- “Control Structure testing strategies”: named, but exact strategy list not provided.
- “Thread based testing” and “Use-Based testing”: named, but not defined.
- “Framework” and “Hooks”: named, but not explained.
- “Common types of attacks”: named, but no explicit list appears in the extracted term list.
When exact lecture-specific answers are needed, use the actual W6–W13 lecture slides or instructor notes.

============================================================
PART G — INSTRUCTIONS FOR THE LLM USING THIS BANK
============================================================

If administering an oral or interactive test:
1. Ask one question at a time.
2. Do not reveal the answer before the user responds.
3. Mix definition, comparison, diagram interpretation, and scenario questions.
4. Ask follow-up questions when an answer is vague, circular, or missing a key distinction.
5. Require examples for concepts that are easy to memorize without understanding.
6. Track which sections have been tested and which concepts remain untested.
7. Give direct, specific feedback. Identify exactly what was correct, missing, or incorrect.
8. Do not overpraise an answer that is only partially correct.
9. Distinguish a factual error from a missing detail.
10. When a concept is not defined in the supplied review, say so. Do not invent the professor's wording.
11. For CCA, CO, UI golden-rule wording, thread-based testing, use-based testing, hooks, QA elements, and control-structure strategy lists, ask for lecture notes if exact course-specific grading is required.
12. At the end, provide a coverage report listing every topic as Strong, Partial, Weak, or Not Tested.

If generating a written exam from this bank:
- Do not restrict yourself to the professor's original 16-question format unless specifically asked.
- Cover every term and concept.
- Include multiple-choice, short-answer, compare/contrast, diagram interpretation, and scenario-based questions.
- Include an answer key separately from the questions.
- Avoid questions that depend on facts not present in the review unless explicitly labeled as general software-engineering context.
"""
out='/mnt/data/ECEN4273_Exhaustive_LLM_Test_Bank.txt'
pypandoc.convert_text(content, 'plain', format='md', outputfile=out, extra_args=['--standalone'])
print(os.path.getsize(out), len(content.split()), out)
