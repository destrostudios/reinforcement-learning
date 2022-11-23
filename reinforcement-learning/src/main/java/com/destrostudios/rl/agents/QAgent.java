package com.destrostudios.rl.agents;

import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDManager;
import com.destrostudios.rl.Agent;
import com.destrostudios.rl.Environment;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.listener.TrainingListener.BatchData;
import com.destrostudios.rl.EnvironmentStep;
import lombok.AllArgsConstructor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.ConcurrentHashMap;

/**
 * An agent that implements Q or Deep-Q Learning.
 *
 * Deep-Q Learning estimates the total reward that will be given until the environment ends in a  * particular state after taking a particular action.
 * Then, it is trained by ensuring that the prediction before taking the action match what would be predicted after taking the action.
 * More information can be found in this <a href="https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf">paper</a>.
 *
 * It is one of the earliest successful techniques for reinforcement learning with Deep learning.
 * While being a good introduction to the field, many better techniques are commonly used now.
 */
@AllArgsConstructor
public class QAgent implements Agent {

    private static final Logger logger = LoggerFactory.getLogger(QAgent.class);

    private Trainer trainer;
    // The reward discount to apply to rewards from future states
    private float rewardDiscount;

    @Override
    public NDList chooseAction(Environment environment, boolean isTraining) {
        ArrayList<NDList> actionSpace = environment.getActionSpace();
        NDArray actionReward = trainer.evaluate(environment.getCurrentObservation()).singletonOrThrow().get(0);
        logger.info(Arrays.toString(actionReward.toFloatArray()));
        int bestAction = Math.toIntExact(actionReward.argMax().getLong());
        return actionSpace.get(bestAction);
    }

    @Override
    public void train(EnvironmentStep[] batchSteps) {
        BatchData batchData = new BatchData(null, new ConcurrentHashMap<>(), new ConcurrentHashMap<>());

        // Temporary manager for attaching NDArray to reduce the gpu memory usage
        NDManager temporaryManager = NDManager.newBaseManager();

        NDList preObservationBatch = new NDList();
        for (EnvironmentStep step : batchSteps) {
            NDList preObservation = step.getPreObservation();
            preObservation.attach(temporaryManager);
            preObservationBatch.addAll(preObservation);
        }
        NDList preInput = new NDList(NDArrays.concat(preObservationBatch, 0));

        NDList postObservationBatch = new NDList();
        for (EnvironmentStep step : batchSteps) {
            NDList postObservation = step.getPostObservation();
            postObservation.attach(temporaryManager);
            postObservationBatch.addAll(postObservation);
        }
        NDList postInput = new NDList(NDArrays.concat(postObservationBatch, 0));

        NDList actionBatch = new NDList();
        Arrays.stream(batchSteps).forEach(step -> actionBatch.addAll(step.getAction()));
        NDList actionInput = new NDList(NDArrays.stack(actionBatch, 0));

        NDList rewardBatch = new NDList();
        Arrays.stream(batchSteps).forEach(step -> rewardBatch.addAll(new NDList(step.getReward())));
        NDList rewardInput = new NDList(NDArrays.stack(rewardBatch, 0));

        try (GradientCollector collector = trainer.newGradientCollector()) {
            NDList QReward = trainer.forward(preInput);
            NDList targetQReward = trainer.forward(postInput);

            NDList Q = new NDList(QReward.singletonOrThrow()
                    .mul(actionInput.singletonOrThrow())
                    .sum(new int[]{1}));

            NDArray[] targetQValue = new NDArray[batchSteps.length];
            for (int i = 0; i < batchSteps.length; i++) {
                if (batchSteps[i].isTerminal()) {
                    targetQValue[i] = batchSteps[i].getReward();
                } else {
                    targetQValue[i] = targetQReward.singletonOrThrow().get(i)
                            .max()
                            .mul(rewardDiscount)
                            .add(rewardInput.singletonOrThrow().get(i));
                }
            }
            NDList targetQBatch = new NDList();
            Arrays.stream(targetQValue).forEach(value -> targetQBatch.addAll(new NDList(value)));
            NDList targetQ = new NDList(NDArrays.stack(targetQBatch, 0));

            NDArray lossValue = trainer.getLoss().evaluate(targetQ, Q);
            collector.backward(lossValue);
            batchData.getLabels().put(targetQ.singletonOrThrow().getDevice(), targetQ);
            batchData.getPredictions().put(Q.singletonOrThrow().getDevice(), Q);
            trainer.step();
        }
        for (EnvironmentStep step : batchSteps) {
            step.getPreObservation().attach(step.getManager());
            step.getPostObservation().attach(step.getManager());
        }
        temporaryManager.close();
    }
}
