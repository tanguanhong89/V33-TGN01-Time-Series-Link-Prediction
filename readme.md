# Time Series Edge probability
## Problem
How to find anomaly in a problem where events are spawning in incremental time, represented by layers of family tree graphs overlaying each other, (and possibly speckled with orphan events), expressed as follows?

![N|Solid](https://raw.githubusercontent.com/PhantomV1989/V33-TGN01-Time-Series-Link-Prediction/master/src/problem.png)
Assumption: 
    - Each event has at most 1 parent
    - Each child will always spawn after its parent’s time of occurence
    
## Solution
Given the above problem, we can approach it in 2 ways:
1) Model the graph structure of the problem, then find anomalies
	- Pros:  
    • Able to see the more complete picture of the overall problem
    • Can be possibly used to generate ‘healthy’ and ‘unhealthy’ graphs
    - Cons:  
     • The model is much harder to use/design (subgraphs with spartial & temporal contexts), additional performance compared to only modelling possible edges may not be guranteed.
2) Model only the edge probabilities.
	- Pros:  
    • Easy to design, train and use
	- Cons:  
    • Very difficult to use in a generative manner, since this approach is specifically designed to model induced probability of existing links (more for validation purposes)

The primary need for this problem is more of to find anomalies in an efficient manner. Although both approaches can detect anomalies, the 2nd approach presents a simpler solution to the problem, hence the it is chosen.

## Edge probability approach
Hence, we turn the above graph problem into a edge probability problem.
Given that event B is a child of event A, what are the chances that this relationship is a legitimate one? (Edge probability)

![N|Solid](https://raw.githubusercontent.com/PhantomV1989/V33-TGN01-Time-Series-Link-Prediction/master/src/fig2.png)

A: Must spawn B given a set of possible circumstances (healthy spawning conditions)
B: Must make sure that the delivery period between A to B, nothing abnormal has happened (child swap, sudden change in birth environment such that existance of B is impossible)

The next question is, how do we determine contextual information for each event A, B? There are many ways to model contextual information but for this case, we are modelling the sequential nature of past events that occured before a particular event as that event’s context.

Using data from Fig.1, we compress everything on to a single timeline.

![N|Solid](https://raw.githubusercontent.com/PhantomV1989/V33-TGN01-Time-Series-Link-Prediction/master/src/fig3.png)

From Fig.3 (or Fig.1) we have Event N as our latest event, and its parent is event M, which occured 5~6 events prior.

The context information for events M, N can be intepreted as follows:

![N|Solid](https://raw.githubusercontent.com/PhantomV1989/V33-TGN01-Time-Series-Link-Prediction/master/src/fig4.png)

Do note that because the model itself uses LSTM, which uses prior cell state for inference (means that it takes into account the whole history if possible), and that M and N shares the same environment, hence we see here that part of N’s context is actually M’s context.(sM is subset of sN if both are in the same environment, which is usually the case)

Of course, one could argue why not just use a single LSTM for sN to predict existance of N (P(N)=f(sN)) since sN contains information from sM? Because ignoring parental information, even if conditions for birth are ideal, can result in false prediction.

## Design Consideration I: Time
In this model, time relative to the child event is used.  The value is normalized using tanh to model the latest event’s time sensitiveness to other events.

![N|Solid](https://raw.githubusercontent.com/PhantomV1989/V33-TGN01-Time-Series-Link-Prediction/master/src/fig5.png)

![N|Solid](https://raw.githubusercontent.com/PhantomV1989/V33-TGN01-Time-Series-Link-Prediction/master/src/timetanh.png)
where 
    - X is any event
    - k is a scalar to determine the period of time event N should be sensitive to. Smaller number means more sensitive to longer periods of time
    - t*<sub>XN</sub> = t<sub>X</sub>-t<sub>N</sub> (time of event X – time of event N, if X is past event, this value is negative)
    - tanh is hyperbolic tangent

## Design Consideration II: Other event properties
An event may have other properties, which may influence the outcome of the model. To be as comprehesive as possible, these properties are stacked together with the event’s embeddings as part of training inputs.
# Overall design
## LSTM layer
With reference to latest event N, propagate LSTM from 1st event to event M (parent of N) to get s<sub>M</sub>, h<sub>M</sub>
