package com.destrostudios.rl.training;

import ai.djl.Model;
import ai.djl.modality.rl.agent.QAgent;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import com.destrostudios.rl.Environment;
import lombok.AllArgsConstructor;

@AllArgsConstructor
public class Tester {

    private Environment environment;
    private TrainerConfig config;

    public void test(Model model) {
        NDManager baseManager = NDManager.newBaseManager();
        TrainerEnvironment env = new TrainerEnvironment(baseManager, environment, config);
        env.setTrainSubManager(baseManager.newSubManager());
        try (ai.djl.training.Trainer trainer = model.newTrainer(config.getTrainingConfig())) {
            QAgent agent = new QAgent(trainer, config.getRewardDiscount());
            while (true) {
                NDList action = agent.chooseAction(env, false);
                environment.takeAction(action);
            }
        }
    }
}
