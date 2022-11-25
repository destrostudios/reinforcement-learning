package com.destrostudios.rl.training;

import ai.djl.Model;
import ai.djl.ndarray.NDList;
import com.destrostudios.rl.agents.QAgent;
import com.destrostudios.rl.Agent;
import com.destrostudios.rl.Environment;
import lombok.AllArgsConstructor;

@AllArgsConstructor
public class Tester {

    private Environment environment;
    private TrainerConfig config;

    public void test(Model model) {
        try (ai.djl.training.Trainer trainer = model.newTrainer(config.getTrainingConfig())) {
            Agent agent = new QAgent(trainer, config.getRewardDiscount());
            while (true) {
                NDList action = agent.chooseAction(environment, false);
                environment.takeAction(action);
            }
        }
    }
}
