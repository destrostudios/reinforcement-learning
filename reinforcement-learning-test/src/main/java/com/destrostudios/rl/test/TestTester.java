package com.destrostudios.rl.test;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import com.destrostudios.rl.training.Tester;
import com.destrostudios.rl.test.game.FlappyBird;
import com.destrostudios.rl.training.TrainerConfig;

import java.io.IOException;

public class TestTester {

    public static void main(String[] args) throws IOException, MalformedModelException {
        FlappyBird environment = new FlappyBird(true);
        Model model = TestModelLoader.loadModel();
        Tester tester = new Tester(environment, new TrainerConfig());
        tester.test(model);
    }
}
