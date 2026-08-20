# Terraflux (涌陆): A Board Game Where the Board Plays Back

**Three breaks from board-game tradition — and what self-play reinforcement learning revealed about each**

*AJiTai · August 2026*
*Code: [github.com/ajt071220-netizen/terraflux](https://github.com/ajt071220-netizen/terraflux) · Playable demo: [ajt071220-netizen.github.io/terraflux](https://ajt071220-netizen.github.io/terraflux/) · Archive: [doi.org/10.5281/zenodo.16898141](https://doi.org/10.5281/zenodo.16898141)*

## Abstract

I designed Terraflux (涌陆), an asymmetric board game in which the board itself takes part in the play, alongside the pieces. It breaks from traditional board games in three ways: the fluid mutation of the board's terrain, a terrain-owned height rule, and asymmetric pursuit-evasion goals. I trained an AI referee through self-play reinforcement learning (PPO) to examine, one by one, the new questions these three designs raise. There are three findings. First, the terrain mutation leaves a non-stationary signature on the training curves: the win rate never converges, and the value function never saturates. Second, under the long causal chain of height-based suppression, sparse rewards prove viable — but the win rates recorded in self-play logs are not universally comparable; the truth has to be settled by head-to-head round-robin play. Finally, fairness is not an inherent property of a layout: in one and the same opening, White's win rate climbs with policy strength from 0% to 46%.

## 1. Introduction — Boards That Don't Sit Still

In the overwhelming majority of board games played around the world today, the board is a silent actor. Moves may vary infinitely, but the stage itself never changes. What, then, if the board itself joined the play? With this thought, I designed Terraflux (涌陆), a board game on which I set out to test three breaks from tradition.

Break one: every move rewrites the terrain around its landing cell — the board is no longer static. Break two: height belongs to the board, not the pieces; capture no longer relies on the traditional displacement of moving onto the opponent's cell, but on commanding height. Break three: the two sides' goals are no longer symmetric. White is a fugitive, fleeing toward a designated victory edge in the south; Black is the pursuer. On top of these come two alternating terrain phases governed by odd and even rounds, and a draw in which both sides satisfy their winning conditions at the same adjudication instant.

Each break raises a question no traditional board game has ever had to answer. On a board without a stable state, what does a reinforcement-learned policy look like? When height decides victory and defeat, is a sparse terminal reward enough? And for an asymmetric game, who is qualified to judge its fairness? I trained a self-play PPO agent as the referee. This report puts each of the three broken conventions on trial before that RL referee, and observes what happens.

## 2. The Game in One Page

Terraflux is played on a 25×25 board of **pillars** (位柱) — the standing columns whose heights the terrain continually rewrites. White starts from the northern half and wins by reaching the designated victory edge in the south; Black wins by capture — standing adjacent to White on a strictly higher pillar. Players alternate turns, with Black moving first by default. Every move rewrites the terrain around its landing cell: on odd-numbered rounds the cross phase fires (the pillar to the north rises, the one to the south sinks, and those to the east and west are filled solid, becoming impassable); on even-numbered rounds the X phase fires, raising and lowering pillars along the diagonals, and so the cycle continues. Pillars cycle through three states — raised, lowered, filled — and filling is never final: a later phase can always reactivate a filled pillar. Beyond winning and losing there is a third outcome: a draw, triggered when both sides satisfy their respective winning conditions at the same instant. (Other terminal cases — threefold repetition, and a player left with no legal move being declared the loser — are documented in the repository.) The project ships with three kinds of AI agents — a heuristic, an LLM hybrid, and a PPO policy — and a bilingual (Chinese/English) 3D online simulator, in which visitors can watch a narrated demo game in the browser.

![Figure 1](../experiments/fig_size_fairness.png)

*Figure 1. Board size is not the fairness variable: white win rate stays flat across seven board sizes from 15×15 to 27×27 (heuristic self-play, duel gap=3).*

**Table 1. Terraflux in the design space of classic abstract games**

| Dimension | Go / Chess | Zertz | Dvonn | **Terraflux** |
|---|---|---|---|---|
| Board | Static forever | Shrinks (tiles removed) | Static | **Rewritten by every move** |
| Height | — | — | Stacked pieces | **Terrain lifts the piece** |
| Objective | Symmetric | Symmetric | Symmetric | **Asymmetric (flee vs. hunt)** |
| Capture | Move onto piece | Jump over | Surround | **Adjacent + strictly higher** |

## 3. Manufacturing the Referee

A newly invented game has no human player community, so the question of its fairness can only be answered by manufacturing referees. I implemented the engine twice: a JavaScript version drives the Three.js 3D front end, and a line-by-line Python port drives RL training, with the two implementations cross-validating every rule. Training uses MaskablePPO with invalid-action masking: the observation is a 25×25×4 one-hot encoding of the terrain plus both piece positions and 8 dimensions of metadata; the action space is 8 directions plus a pass; the reward is ±1 at game end (a control group adds a −0.002 step penalty per move). The opponent pool mixes 65% heuristic AI with 35% of the 5 most recent historical snapshots, preventing the policy from overfitting to a single opponent; main training runs 500k steps. The evaluation protocol has two tracks — self-play win rate, and direct round-robin play between models — with 200–300 games per condition and 95% confidence intervals reported. One discipline governs everything: for cross-model strength comparisons, the round-robin data is the only source of truth. Section 5 will prove this discipline necessary.

## 4. Break I — The Board Plays Back

In the overwhelming majority of board games worldwide, the board does not participate in the contest; it merely serves as a container for the pieces. The few boards that do take part typically change in only one direction — once a cell is gone, it never comes back. The terrain of Terraflux is fluid: raised, lowered, and filled states cycle, and a filled pillar can be restored by a later phase. In play, this means there is no eternally optimal position — what is safe today may collapse in three moves; what is a dead end today may be reopened by the terrain. That raises a question worth discussing: on a board with no stable state, what does a learned policy look like? Can training converge at all?

**The RL verdict.** The rollout-level logs of the 500k-step main training run give the answer: it does not converge — and it should not. After the win rate climbs from chance level into the 0.87 range, the mid-phase of training (150k–400k steps) enters sustained, wide oscillation that never settles, even by the end of training. The oscillation is not an experimental error but the signature of dual non-stationarity: the terrain is rewritten every single move, and the opponent pool is populated by the learner's own historical snapshots — as the learner grows stronger, so do its opponents, in lockstep. Win-rate tug-of-war is the natural state of a co-evolving system. Every diagnostic points to healthy training: no KL spikes (threshold 0.15) anywhere in the run; the only two entropy jumps (at 94k and 104k) land exactly on snapshot-injection moments; and the value function's explained variance rises from 0.2 to 0.8 without ever saturating — proof that both opponent and board kept changing, and that the question "will this game be won?" never had a stable answer. This is the fingerprint left on the training curves by a board that participates in the play: the effect of fluid terrain is not confined to one corner or another — it reshapes the learning problem at its root.

**Table 2. Training phases on a fluid board**

| Phase | Steps | Rolling win rate | Policy entropy | Episode length |
|---|---|---|---|---|
| Early | 2k–165k | 0.419 | −1.119 | 23.1 |
| Mid | 167k–331k | 0.672 | −0.850 | 25.8 |
| Late | 333k–501k | 0.712 | −0.757 | 20.0 |

![Figure 2](../experiments/fig_training_dynamics.png)

*Figure 2. Training dynamics: win rate, entropy, episode length, explained variance.*

## 5. Break II — The Terrain Owns the Height

In the vast majority of traditional board games, height is an attribute of the piece — its rank, or its weight in the game. In Terraflux, height belongs to the board itself: a spherical piece standing on a dominant pillar and one standing on a sunken pillar face entirely different winning conditions. Out of this mechanism grows Black's path to victory: only from a position of height advantage can Black hunt White down. At the level of the rules, terrain advantage is itself a weapon, and the board's height differences directly shape the play. For example, if Black stands on a raised pillar (+1) and White on a lowered one (−1), the height gap is 2. This raises a question: building a height advantage and then completing a capture is a long causal chain with no significant reward signal in between. With only a ±1 terminal score as a sparse reward, this is clearly a test for PPO.

**The RL verdict.** I ran a controlled experiment: two PPO groups trained from scratch, the only variable being the step penalty. In the first group, each move costs −0.002; in the other, the step penalty is 0 (both groups share the same ±1 terminal reward). Each group trained for 500k steps. Two important answers came out.

Answer one: sparse-reward training is viable — just not as fast. To reach a 55% rolling win rate, the pure sparse group needed 73k steps, while the step-penalty group needed only 59k — the former about 24% slower; at the 65% threshold, 18% slower.

Answer two is the more important one: the win rates written into self-play logs are not to be taken at face value. At the end of training, the two groups' logged win rates were nearly identical — 71.6% versus 70.9%. But in actual head-to-head play between the two trained models, the result was 66%:14.5%, a clear advantage for the step-penalty group. This shows that a logged win rate is a relative tally recorded against a particular opponent pool, not a universal, absolute measure of strength.

**Table 3. Sparse reward suffices — but self-play logs lie**

| Metric | Shaped | Sparse |
|---|---|---|
| Steps to rolling win rate 55% | **59k** | 73k (+24% slower) |
| Steps to rolling win rate 65% | **65k** | 77k (+18% slower) |
| Late-stage win rate vs own snapshot pool | 0.716 | 0.709 (looks tied) |
| **Head-to-head round robin** | **66.0%** | 14.5% (19.5% draws) |
| Mean game length (late stage) | 28.8 | 29.2 |

![Figure 3](../experiments/fig_ablation.png)

*Figure 3. Reward shaping ablation: win rate / episode length / policy entropy.*

## 6. Break III — Asymmetric Goals, Rotating Phases, and the Same-Instant Draw

In the most familiar board games, such as Go and chess, the two sides' goals are mirror images of each other: identical pieces attack one another in identical ways. Terraflux is fundamentally asymmetric. The white side is a fugitive that must cross the entire board to reach a designated edge; the black side is the pursuer, which must keep pace with White along its escape route and find a height advantage to win. In how the game is won, this is entirely different from Go and chess.

The shape of each terrain rewrite is not fixed either, but alternates between two cases: on odd-numbered triggers, a cross-shaped terrain change fires (the north rises, the south sinks, and the east and west are filled into impassable ground); on even-numbered triggers, it becomes the surrounding X shape. And so the cycle goes.

This asymmetric rule of engagement has also given birth to something almost nonexistent in other board games: the same-instant draw. It occurs when, at a single adjudication moment, both sides satisfy their winning conditions — White reaches the designated edge, while Black happens to be right behind White in a position of height advantage. The trigger is White's own move: the move rewrites the surrounding terrain once more and happens to lift Black onto a geographic high ground, producing exactly this outcome.

**The RL verdict.** Yet this design naturally raises a question — the fairness of the game — which can only be judged and improved through experiment and data. The first step was to manufacture referees: four policies of different strengths playing against themselves under an identical opening (Black in the exact center of the board, White three pillars due north of Black, Black moving first).

The curve formed by this data is as follows. In the weakest matchup — heuristic versus itself — White's win rate is 0%: not a single game won, with capture arriving after about 10 moves on average. A PPO trained on sparse rewards reaches 20%; the step-penalty PPO reaches 34%; and the main PPO, trained for 500k steps, reaches 46% — strictly monotonic, gradually approaching the perfect balance of 50%. Meanwhile, as the board size was swept from 15×15 up to 27×27 across seven variants, every observed curve came out flat — board size has almost no effect on fairness. Geometry is not the variable; strength is.

Why does the weakest White lose so inevitably? The answer requires dissecting the heuristic's evaluation function. Although it contains a "keep distance" term, that term's weight is completely overwhelmed by the southward, edge-rushing drive — in effect, White cannot execute an escape at all. Black, by contrast, closes in along a straight line, producing a position in which White is certain to lose within about ten moves. A fully trained PPO, however, taught itself an entirely different method: keeping its distance, and using the board's terrain height differences to position itself sensibly. So the outcome depends greatly on the stock of skills the white side has learned — the more it has learned, the closer the opening moves toward the fairness line.

Therefore, the fairness of a game admits no answer detached from its players. Fairness is not an inherent property of a layout, but a spectrum that climbs with the strength of the referee. For asymmetric games of this kind, only a sufficiently strong policy is qualified to serve as the judge of fairness; and whether phase rotation and the same-instant draw are themselves balanced can, from now on, be measured rather than argued. This, in turn, contributes valuable phenomena and data to the later improvement and adjustment of the game's own rules.

**Table 4. White win rate rises with policy strength**

| Policy | Round-robin rank | White win rate (duel self-play) | Mean length |
|---|---|---|---|
| Heuristic vs itself | 4 (weakest) | **0%** | 10 |
| PPO sparse | 3 | **20%** (95% CI ±5.6%) | 59 |
| PPO shaped | 2 | **34%** (±6.5%) | 75 |
| PPO main (500k steps) | 1 (strongest) | **46%** (±6.9%) | 54 |

![Figure 4](../experiments/fig_strength_fairness.png)

*Figure 4. Left: white win rate rises monotonically with policy strength toward 50%. Right: round-robin strength matrix.*

## 7. Discussion — The Mirror-Value Test, Limitations, and Future Mechanisms

Looking back at this data, one design discipline runs behind all three breaks. I call it the **mirror-value test**: in an asymmetric pursuit-evasion game, every word in the terrain vocabulary must be roughly symmetric in value between the fleeing side and the hunting side. Among the candidate mechanisms, teleportation benefits only the fugitive and freezing benefits only the pursuer — one-sided weapons are all unfair and disqualified. In the end, only two things passed the test: height and passability. This discipline guarantees that the terrain remains a neutral condition, not one tilted toward either side.

The limitations are stated honestly, four of them: all training used a single random seed; the weak-policy end is represented by only one kind of heuristic; the full sweep of strong policies across many board sizes could not be completed due to limited compute; and each condition ran 200–300 games, giving a 95% confidence interval of about ±6% — finer differences would require more games.

Future work is a mechanism pipeline. Eight new candidate mechanisms that claim to pass the mirror-value test are waiting on the list, but each of them must pass an actual test by the RL referee before it is formally admitted. The pipeline consists of: new idea, implementation, measurement, verdict by data — and loop. This looping process itself has also become one of the project's products: an experimental ground where new conditions and variables can be continually added to explore new variations.

## 8. References and Links

- **Archived release (v1.0.0, citable DOI)**: [10.5281/zenodo.16898141](https://doi.org/10.5281/zenodo.16898141)
- **Code & full experiment data**: [github.com/ajt071220-netizen/terraflux](https://github.com/ajt071220-netizen/terraflux)
- **Playable 3D demo (bilingual, with narrated tutorial game)**: [ajt071220-netizen.github.io/terraflux](https://ajt071220-netizen.github.io/terraflux/)
- Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
- Raffin, A. et al. (2021). *Stable-Baselines3: Reliable Reinforcement Learning Implementations*. JMLR 22(268).
- Huang, S. & Ontañón, S. (2022). *A Closer Look at Invalid Action Masking in Policy Gradient Algorithms*. arXiv:2006.14171.
