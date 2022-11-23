package com.destrostudios.rl.training;

import ai.djl.Model;
import ai.djl.ndarray.NDList;
import ai.djl.training.TrainingConfig;
import com.destrostudios.rl.agents.QAgent;
import com.destrostudios.rl.Agent;
import com.destrostudios.rl.Environment;
import lombok.AllArgsConstructor;

@AllArgsConstructor
public class Tester {

    private Environment environment;
    private TrainingConfig config;

    public void test(Model model) {
        try (ai.djl.training.Trainer trainer = model.newTrainer(config)) {
            Agent agent = new QAgent(trainer, Trainer.REWARD_DISCOUNT);
            while (true) {
                NDList action = agent.chooseAction(environment, false);
                environment.takeAction(action);
            }
        }
    }
}
