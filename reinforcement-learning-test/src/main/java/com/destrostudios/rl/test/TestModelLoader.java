package com.destrostudios.rl.test;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import com.destrostudios.rl.test.game.Constant;

import java.io.IOException;
import java.nio.file.Paths;

public class TestModelLoader {

    private static final String MODEL_PATH = Constant.RESOURCE_PATH + "/model";

    public static Model loadModel() throws IOException, MalformedModelException {
        Model model = Model.newInstance("QNetwork");
        model.setBlock(getBlock());
        model.load(Paths.get(MODEL_PATH), "dqn-trained");
        return model;
    }

    private static SequentialBlock getBlock() {
        // conv -> conv -> conv -> fc -> fc
        return new SequentialBlock()
                .add(Conv2d.builder()
                        .setKernelShape(new Shape(8, 8))
                        .optStride(new Shape(4, 4))
                        .optPadding(new Shape(3, 3))
                        .setFilters(4)
                        .build())
                .add(Activation::relu)

                .add(Conv2d.builder()
                        .setKernelShape(new Shape(4, 4))
                        .optStride(new Shape(2, 2))
                        .setFilters(32)
                        .build())
                .add(Activation::relu)

                .add(Conv2d.builder()
                        .setKernelShape(new Shape(3, 3))
                        .optStride(new Shape(1, 1))
                        .setFilters(64)
                        .build())
                .add(Activation::relu)

                .add(Blocks.batchFlattenBlock())

                .add(Linear.builder()
                        .setUnits(512)
                        .build())
                .add(Activation::relu)

                .add(Linear.builder()
                        .setUnits(2)
                        .build());
    }
}
