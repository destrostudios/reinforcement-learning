package com.destrostudios.rl.test.moba;

import lombok.Getter;
import org.deeplearning4j.rl4j.environment.*;
import org.nd4j.linalg.api.rng.Random;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class MobaEnv implements Environment<Integer> {

    public MobaEnv(Random random) {
        schema = new Schema<>(new ValidIntegerActionSchema(ACTIONS_COUNT, 0, random, this));
    }
    /*
    -> do nothing
    0

    -> walk
    1,2,3,4

    -> attack
    5,6,7
    8,9,10
    11,12,13
    */
    public static final int ACTIONS_COUNT = 1 + 4 + 9;
    @Getter
    private Schema<Integer> schema;
    @Getter
    private MobaMap map;

    @Override
    public Map<String, Object> reset() {
        map = new MobaMap();
        return getChannelsData();
    }

    @Override
    public boolean isValidAction(Integer action) {
        if (action != 0) {
            MobaObject player = map.getPlayer();
            int remainingActionIndex = action - 1;
            if (remainingActionIndex < 4) {
                switch (remainingActionIndex) {
                    case 0: return (player.getX() > 0);
                    case 1: return (player.getX() < (MobaMap.WIDTH - 1));
                    case 2: return (player.getY() > 0);
                    case 3: return (player.getY() < (MobaMap.HEIGHT - 1));
                }
            } else {
                remainingActionIndex -= 4;
                int targetX = player.getX() + ((remainingActionIndex % 3) - 1);
                int targetY = player.getY() + ((remainingActionIndex / 3) - 1);
                MobaObject target = map.getObject(targetX, targetY);
                return ((target != null) && player.canAttack(target));
            }
        }
        return true;
    }

    @Override
    public StepResult step(Integer action) {
        if (!isValidAction(action)) {
            System.out.println("?");
        }
        MobaObject player = map.getPlayer();
        int team1HealthOld = map.getTeamHealth(1);
        int team2HealthOld = map.getTeamHealth(-1);
        double reward = 0;
        if (action != 0) {
            int remainingActionIndex = action - 1;
            if (remainingActionIndex < 4) {
                switch (remainingActionIndex) {
                    case 0:
                        player.moveLeft();
                        break;
                    case 1:
                        player.moveRight();
                        break;
                    case 2:
                        player.moveUp();
                        break;
                    case 3:
                        player.moveDown();
                        break;
                }
            } else {
                remainingActionIndex -= 4;
                int targetX = player.getX() + ((remainingActionIndex % 3) - 1);
                int targetY = player.getY() + ((remainingActionIndex / 3) - 1);
                player.attack(targetX, targetY);
                int team2HealthNew = map.getTeamHealth(-1);
                reward += ((team2HealthOld - team2HealthNew) * 20);
            }
        }
        map.nextFrame();
        if ((map.getFrame() % 1) == 0) {
            System.out.println(map.getAsciiImage());
        }
        reward += getTeamHealthChangeReward(team1HealthOld, team2HealthOld);
        if (player.isDead() || map.getTower1().isDead()) {
            reward -= 1000;
        }
        if (map.getTower2().isDead()) {
            reward += 1000;
        }
        return new StepResult(getChannelsData(), reward, isEpisodeFinished());
    }

    private double getTeamHealthChangeReward(int team1HealthOld, int team2HealthOld) {
        int team1HealthNew = map.getTeamHealth(1);
        int team2HealthNew = map.getTeamHealth(-1);
        return ((team1HealthNew - team1HealthOld) + (team2HealthOld - team2HealthNew));
    }

    private Map<String, Object> getChannelsData() {
        MobaObject player = map.getPlayer();
        int nearestObjectsLimit = 5;
        double[] mydata = new double[4 + (nearestObjectsLimit * 5)];
        int index = 0;
        mydata[index++] = player.getX();
        mydata[index++] = player.getY();
        mydata[index++] = player.getHealth();
        mydata[index++] = player.getRemainingAttackCooldown();
        List<MobaObject> objects = player.getNearestObjects(nearestObjectsLimit);
        for (int i = 0; i < nearestObjectsLimit; i++) {
            MobaObject object = ((i < objects.size()) ? objects.get(i) : null);
            if (object != null) {
                mydata[index++] = object.getTeam();
                mydata[index++] = player.getDistanceX(object.getX());
                mydata[index++] = player.getDistanceY(object.getY());
                mydata[index++] = object.getHealth();
                mydata[index++] = object.getRemainingAttackCooldown();
            } else {
                index += 5;
            }
        }
        HashMap<String, Object> channelsData = new HashMap<>();
        channelsData.put("mydata", mydata);
        return channelsData;
    }

    @Override
    public boolean isEpisodeFinished() {
        return map.getTower1().isDead() || map.getTower2().isDead();
    }

    @Override
    public void close() {
        // Not needed
    }
}
