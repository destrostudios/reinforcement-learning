/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.deeplearning4j.rl4j.agent.learning.behavior;

import org.deeplearning4j.rl4j.environment.action.Action;
import org.deeplearning4j.rl4j.environment.observation.Observation;

public interface LearningBehavior<OBSERVATION extends Observation, ACTION extends Action> {

    /**
     * This method is called when a new episode has been started.
     */
    void handleEpisodeStart();

    /**
     * This method is called when new experience is generated.
     *
     * @param observation The observation prior to taking the action
     * @param action The action that has been taken
     * @param reward The reward received by taking the action
     * @param isTerminal True if the episode ended after taking the action
     */
    void handleNewExperience(OBSERVATION observation, ACTION action, double reward, boolean isTerminal);

    /**
     * This method is called when the episode ends or the maximum number of episode steps is reached.
     *
     * @param finalObservation The observation after the last action of the episode has been taken.
     */
    void handleEpisodeEnd(OBSERVATION finalObservation);

    /**
     * Notify the learning behavior that a step will be taken.
     */
    void notifyBeforeStep();
}
