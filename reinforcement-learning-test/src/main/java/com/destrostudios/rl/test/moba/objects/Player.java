package com.destrostudios.rl.test.moba.objects;

import com.destrostudios.rl.test.moba.MobaObject;

public class Player extends MobaObject {

    public Player(int team, int spawnX, int spawnY) {
        super(team, spawnX, spawnY, 20, 2, 1, 2);
        this.spawnX = spawnX;
        this.spawnY = spawnY;
    }
    private int spawnX;
    private int spawnY;

    @Override
    protected void onDeath() {
        super.onDeath();
        x = spawnX;
        y = spawnY;
        setToMaximumHealth();
    }

    @Override
    protected String getAsciiLetter() {
        return "P";
    }
}
