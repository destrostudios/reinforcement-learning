package com.destrostudios.rl.test.moba.objects;

import com.destrostudios.rl.test.moba.MobaMap;
import com.destrostudios.rl.test.moba.MobaObject;

public class Player extends MobaObject {

    public Player(int team, int x, int y) {
        super(team, x, y, 20, 2, ATTACK_RANGE, 2);
    }
    public static final int ATTACK_RANGE = 2;

    @Override
    protected void onDeath() {
        super.onDeath();
        x = MobaMap.PLAYER_SPAWN_X;
        y = MobaMap.PLAYER_SPAWN_Y;
        setToMaximumHealth();
    }

    @Override
    protected String getAsciiLetter() {
        return "P";
    }
}
