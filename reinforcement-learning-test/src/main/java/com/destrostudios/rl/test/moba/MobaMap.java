package com.destrostudios.rl.test.moba;

import com.destrostudios.rl.test.moba.objects.Minion;
import com.destrostudios.rl.test.moba.objects.Player;
import com.destrostudios.rl.test.moba.objects.Tower;
import lombok.Getter;

import java.util.LinkedList;

/*
P-------------
--------------
T-----MM-----T
--------------
--------------
S = Spawn
T = Tower
M = Minion
P = Player
*/
public class MobaMap {

    public static final int WIDTH = 14;
    public static final int HEIGHT = 5;
    public static final int LANE_Y = 2;
    public static final int PLAYER_SPAWN_X = 0;
    public static final int PLAYER_SPAWN_Y = 0;
    public static final int MINION_SPAWN_INTERVAL = 20;
    @Getter
    private int frame;
    @Getter
    private LinkedList<MobaObject> objects = new LinkedList<>();
    @Getter
    private MobaObject player = new Player(1, PLAYER_SPAWN_X, PLAYER_SPAWN_Y);
    @Getter
    private MobaObject tower1 = new Tower(1, 0, LANE_Y);
    @Getter
    private MobaObject tower2 = new Tower(-1, WIDTH - 1, LANE_Y);

    public MobaMap() {
        add(player);
        add(tower1);
        add(tower2);
    }

    public void nextFrame() {
        frame++;
        if ((frame % MINION_SPAWN_INTERVAL) == 0) {
            trySpawnMinion(1);
            trySpawnMinion(-1);
        }
        for (MobaObject object : objects.toArray(new MobaObject[0])) {
            if (!object.isDead()) {
                object.nextFrame();
            }
        }
    }

    private void trySpawnMinion(int team) {
        int x = ((team == 1) ? tower1.getX() : tower2.getX()) + team;
        int y = LANE_Y;
        if (isFree(x, y)) {
            add(new Minion(team, x, y));
        }
    }

    public void add(MobaObject object) {
        object.setMap(this);
        objects.add(object);
    }

    public void remove(MobaObject object) {
        object.setMap(null);
        objects.remove(object);
    }

    public boolean isFree(int x, int y) {
        return ((x >= 0) && (x < WIDTH) && (y >= 0) && (y < HEIGHT) && (getObject(x, y) == null));
    }

    public MobaObject getObject(int x, int y) {
        return objects.stream().filter(object -> (object.getX() == x) && (object.getY() == y)).findFirst().orElse(null);
    }

    public int getTeamHealth(int team) {
        return objects.stream()
                .filter(object -> object.getTeam() == team)
                .map(MobaObject::getHealth)
                .reduce(0, Integer::sum);
    }

    public String getAsciiImage() {
        String text = "---------- Frame "+  frame + " ----------\n";
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                MobaObject object = getObject(x, y);
                text += ((object != null) ? object.getAsciiImage() : "  --  ");
            }
            text += "\n";
        }
        return text;
    }
}
