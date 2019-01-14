# Time Series Edge probability
## Problem
How to find anomaly in a problem where events are spawning in incremental time, represented by layers of family tree graphs overlaying each other, (and possibly speckled with orphan events), expressed as follows?

![fig1](/src/problem.png)

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

![fig2](/src/fig4.png)

A: Must spawn B given a set of possible circumstances (healthy spawning conditions)
B: Must make sure that the delivery period between A to B, nothing abnormal has happened (child swap, sudden change in birth environment such that existance of B is impossible)

The next question is, how do we determine contextual information for each event A, B? There are many ways to model contextual information but for this case, we are modelling the sequential nature of past events that occured before a particular event as that event’s context.

Using data from Fig.1, we compress everything on to a single timeline.

![fig3](/src/fig3.png)

From Fig.3 (or Fig.1) we have Event N as our latest event, and its parent is event M, which occured 5~6 events prior.

The context information for events M, N can be intepreted as follows:

![fig4](/src/fig4.png)

Do note that because the model itself uses LSTM, which uses prior cell state for inference (means that it takes into account the whole history if possible), and that M and N shares the same environment, hence we see here that part of N’s context is actually M’s context.(sM is subset of sN if both are in the same environment, which is usually the case)

Of course, one could argue why not just use a single LSTM for sN to predict existance of N (P(N)=f(sN)) since sN contains information from sM? Because ignoring parental information, even if conditions for birth are ideal, can result in false prediction.

## Design Consideration I: Time
In this model, time relative to the child event is used.  The value is normalized using tanh to model the latest event’s time sensitiveness to other events.

![fig5](/src/fig5.png)

![N|Solid](/src/timetanh.png)

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

![fig5](/src/fig6.png)

Propagate the same LSTM from 1st event to event N (latest event) to get s<sub>N</sub>, h<sub>N</sub>.

## Fully Connected layer: Trained by binary cross entropy

![fig5](/src/fig7.png)

The output is a single valued number that determines edge probability. 
This is in contrast to conventional classification where one hot encoding is used to possibly represent, in this case, the parent class such that the parent of event N is determined by the greatest value in the encoding array.
The parent-child relationship is intricately modelled by the fully connected layer so one hot encoding model is not needed here.
One benefit of modelling this way is that confidence can be directly determined, while one hot encoding may only tell us which class has the highest probability of being the parent of child N, which may not be  suitable for the needs of this problem.

## Optimization I: Negative training
 To prevent the model from training all inputs to output 100% edge probability, negative training is needed.
Noises will be introduced into the dataset as anomalies for training. The amount of noises can be controlled to train the model sensivity. 
Three different types of noises are used:
### 1) Wrong parent

![fig5](/src/wrongparent.png)

Wrong parent negative training basically means that instead of using an event’s actual parent, we change it to other event types that are not the parent of that event, and train the model against it, 
P(<sub>Enonparent->N</sub>)=0
This is easy to train because all events have at most 1 parent, hence, easy for model to match the correct parent to the current event.

### 2)Event mutation (Insertion & Deletion)
This noise can be thought of as changing an event in a historical context to something else that should not be occuring in a normal behaviour. 

![fig5](/src/fig8.png)

### 3) Event shift
This noise shifts a randomly chosen event along the time dimension such that it is no longer at its original position. 

![fig5](/src/fig9.png)

There are 2 parameters to control this noise, the % historical data to shift (noise), and the shift bracket range. A larger shift bracket range means an event can be shifted further from its original position, hence easier to be identified as an anomaly.

### 4) Others (not implemented)
- Adding random events
- Adding noise to other parameters

**NOTE: It is impossible to incorporate all different possible combination of noises, hence, the model is biased against the type of noise it is trained against. But because the data used for training is generic, hopefully when applied in the real domain, the patterns of actual negative data will fit one of the negative training profile used for training of the model.**

## Optimization II: Loss function
Tanh->Softmax->Binary cross entropy
Ground Truth for positive sample:

Ground Truth for negative sample:

There are currently 3 types of negative noises used for training. For every positive training used, it is matched with a negative training sample chosen at uniform random from any 1 of the 3 noise generators.

Loss: Binary cross entropy

![fig5](/src/bceloss.png)

## Optimization III: Noise training cycle
Because this model trains against noise, the more appropriate way would be to identify the natural noise of the data and generate negative noises that are unlike those so that the model can fit the healthy data better.  However, exactly identifying healthy noises is intractable so uniform noises are used instead. The details of training for 1 data point is as described below: 

![fig5](/src/fig10.png)

Since different levels of noise present different levels of training difficulty for the model, the number of training cycles for each noise level should also reflect that relationship, in particular, lower levels of noise will have higher number of training cycles.
In this model, the scalar multiplier for iteration is a function of time as follows:  
	**K=f(noise)= e(0.35/(noise+0.2))-0.8**
	**Final iteration count for that noise level= K x iteration**
Best weights are saved.

## Optimization IV: White noise event filter
In the actual scenerio, not all events will give good results. This is because many of the events occur frequently in random order, hence have high information entropy (aka useless). We need to differentiate events that can give good results and those that are just white noise.
After noise training, edge probabilities of all nodes in the healthy sample is calculated. Those whose probabilities are too low are kept in a list stored internally so that the model will know when not to make inference for a particular node type since these node types are considered as white noises.

# Possible usecase
Once the model is trained, anomaly detection can be done by chaining a series of high probability edges until a cluster of low probability edges (AUC threshold) or sudden extreme drop in edge probability are encountered.

![fig5](/src/fig11.png)

## Usecase scenerio I: Orphan anomaly
Because this model is trained for edge probabilities, it cannot be used directly for identifying anomalous orphan events. Suppose we have an anomalous event I as shown below:

![fig5](/src/fig12.png)

## Usecase scenerio II: Encountering white noise events
While it is desirable to have each and every node providing us with important information, in the actual scenerio, some of them will be too noisy for meanful intepretations. These nodes are, regretfully, ignored during inference.

![fig5](/src/fig13.png)

# Test results
## To be done

