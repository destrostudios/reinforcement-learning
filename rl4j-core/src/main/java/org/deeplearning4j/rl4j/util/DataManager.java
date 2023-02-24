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

package org.deeplearning4j.rl4j.util;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.Value;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.rl4j.learning.ILearning;
import org.deeplearning4j.rl4j.learning.NeuralNetFetchable;
import org.deeplearning4j.rl4j.learning.configuration.ILearningConfiguration;
import org.deeplearning4j.rl4j.network.ac.IActorCritic;
import org.deeplearning4j.rl4j.network.dqn.DQN;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.common.primitives.Pair;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;

@Slf4j
public class DataManager implements IDataManager {

    final private String home = System.getProperty("user.home");
    final private ObjectMapper mapper = new ObjectMapper();
    private String dataRoot = home + "/" + Constants.DATA_DIR;
    @Getter
    private boolean saveData;
    private String currentDir;

    public DataManager() throws IOException {
        create(dataRoot, false);
    }

    public DataManager(boolean saveData) throws IOException {
        create(dataRoot, saveData);
    }

    public DataManager(String dataRoot, boolean saveData) throws IOException {
        create(dataRoot, saveData);
    }

    private static void writeEntry(InputStream inputStream, ZipOutputStream zipStream) throws IOException {
        byte[] bytes = new byte[1024];
        int bytesRead;
        while ((bytesRead = inputStream.read(bytes)) != -1) {
            zipStream.write(bytes, 0, bytesRead);
        }
    }

    public static void save(String path, ILearning learning) throws IOException {
        try (BufferedOutputStream os = new BufferedOutputStream(new FileOutputStream(path))) {
            save(os, learning);
        }
    }

    public static void save(OutputStream os, ILearning learning) throws IOException {

        try (ZipOutputStream zipfile = new ZipOutputStream(os)) {

            ZipEntry config = new ZipEntry("configuration.json");
            zipfile.putNextEntry(config);
            String json = new ObjectMapper().writeValueAsString(learning.getConfiguration());
            writeEntry(new ByteArrayInputStream(json.getBytes()), zipfile);

            try {
                ZipEntry dqn = new ZipEntry("dqn.bin");
                zipfile.putNextEntry(dqn);

                ByteArrayOutputStream bos = new ByteArrayOutputStream();
                if(learning instanceof NeuralNetFetchable) {
                    ((NeuralNetFetchable)learning).getNeuralNet().save(bos);
                }
                bos.flush();
                bos.close();

                InputStream inputStream = new ByteArrayInputStream(bos.toByteArray());
                writeEntry(inputStream, zipfile);
            } catch (UnsupportedOperationException e) {
                ByteArrayOutputStream bos = new ByteArrayOutputStream();
                ByteArrayOutputStream bos2 = new ByteArrayOutputStream();
                ((IActorCritic)((NeuralNetFetchable)learning).getNeuralNet()).save(bos, bos2);

                bos.flush();
                bos.close();
                InputStream inputStream = new ByteArrayInputStream(bos.toByteArray());
                ZipEntry value = new ZipEntry("value.bin");
                zipfile.putNextEntry(value);
                writeEntry(inputStream, zipfile);

                bos2.flush();
                bos2.close();
                InputStream inputStream2 = new ByteArrayInputStream(bos2.toByteArray());
                ZipEntry policy = new ZipEntry("policy.bin");
                zipfile.putNextEntry(policy);
                writeEntry(inputStream2, zipfile);
            }

            if (learning.getHistoryProcessor() != null) {
                ZipEntry hpconf = new ZipEntry("hpconf.bin");
                zipfile.putNextEntry(hpconf);

                ByteArrayOutputStream bos2 = new ByteArrayOutputStream();
                if(learning instanceof NeuralNetFetchable) {
                    ((NeuralNetFetchable)learning).getNeuralNet().save(bos2);
                }
                bos2.flush();
                bos2.close();

                InputStream inputStream2 = new ByteArrayInputStream(bos2.toByteArray());
                writeEntry(inputStream2, zipfile);
            }


            zipfile.flush();
            zipfile.close();

        }
    }

    public static <C> Pair<IDQN, C> load(File file, Class<C> cClass) throws IOException {
        log.info("Deserializing: " + file.getName());

        C conf = null;
        IDQN dqn = null;
        try (ZipFile zipFile = new ZipFile(file)) {
            ZipEntry config = zipFile.getEntry("configuration.json");
            InputStream stream = zipFile.getInputStream(config);
            BufferedReader reader = new BufferedReader(new InputStreamReader(stream));
            String line = "";
            StringBuilder js = new StringBuilder();
            while ((line = reader.readLine()) != null) {
                js.append(line).append("\n");
            }
            String json = js.toString();

            reader.close();
            stream.close();

            conf = new ObjectMapper().readValue(json, cClass);

            ZipEntry dqnzip = zipFile.getEntry("dqn.bin");
            InputStream dqnstream = zipFile.getInputStream(dqnzip);
            File tmpFile = File.createTempFile("restore", "dqn");
            Files.copy(dqnstream, Paths.get(tmpFile.getAbsolutePath()), StandardCopyOption.REPLACE_EXISTING);
            dqn = new DQN(ModelSerializer.restoreMultiLayerNetwork(tmpFile));
            dqnstream.close();
        }

        return new Pair<IDQN, C>(dqn, conf);
    }

    public static <C> Pair<IDQN, C> load(String path, Class<C> cClass) throws IOException {
        return load(new File(path), cClass);
    }

    public static <C> Pair<IDQN, C> load(InputStream is, Class<C> cClass) throws IOException {
        File tmpFile = File.createTempFile("restore", "learning");
        Files.copy(is, Paths.get(tmpFile.getAbsolutePath()), StandardCopyOption.REPLACE_EXISTING);
        return load(tmpFile, cClass);
    }

    private void create(String dataRoot, boolean saveData) throws IOException {
        this.saveData = saveData;
        this.dataRoot = dataRoot;
        createSubdir();
    }

    //FIXME race condition if you create them at the same time where checking if dir exists is not atomic with the creation
    public String createSubdir() throws IOException {

        if (!saveData)
            return "";

        File dr = new File(dataRoot);
        dr.mkdirs();
        File[] rootChildren = dr.listFiles();

        int i = 1;
        while (childrenExist(rootChildren, i + ""))
            i++;

        File f = new File(dataRoot + "/" + i);
        f.mkdirs();

        currentDir = f.getAbsolutePath();
        log.info("Created training data directory: " + currentDir);

        File mov = new File(getVideoDir());
        mov.mkdirs();

        File model = new File(getModelDir());
        model.mkdirs();

        File stat = new File(getStat());
        File info = new File(getInfo());
        stat.createNewFile();
        info.createNewFile();
        return f.getAbsolutePath();
    }

    public String getVideoDir() {
        return currentDir + "/" + Constants.VIDEO_DIR;
    }

    public String getModelDir() {
        return currentDir + "/" + Constants.MODEL_DIR;
    }

    public String getInfo() {
        return currentDir + "/" + Constants.INFO_FILENAME;
    }

    public String getStat() {
        return currentDir + "/" + Constants.STATISTIC_FILENAME;
    }

    public void appendStat(StatEntry statEntry) throws IOException {

        if (!saveData)
            return;

        Path statPath = Paths.get(getStat());
        String toAppend = toJson(statEntry);
        Files.write(statPath, toAppend.getBytes(), StandardOpenOption.APPEND);

    }

    private String toJson(Object object) throws IOException {
        return mapper.writeValueAsString(object) + "\n";
    }

    public void writeInfo(ILearning iLearning) throws IOException {

        if (!saveData)
            return;

        Path infoPath = Paths.get(getInfo());

        Info info = new Info(iLearning.getClass().getSimpleName(), iLearning.getMdp().getClass().getSimpleName(),
                        iLearning.getConfiguration(), iLearning.getStepCount(), System.currentTimeMillis());
        String toWrite = toJson(info);

        Files.write(infoPath, toWrite.getBytes(), StandardOpenOption.TRUNCATE_EXISTING);
    }

    private boolean childrenExist(File[] files, String children) {
        boolean exists = false;
        for (int i = 0; i < files.length; i++) {
            if (files[i].getName().equals(children)) {
                exists = true;
                break;
            }
        }
        return exists;
    }

    public void save(ILearning learning) throws IOException {

        if (!saveData)
            return;

        save(getModelDir() + "/" + learning.getStepCount() + ".training", learning);
        if(learning instanceof  NeuralNetFetchable) {
            try {
                ((NeuralNetFetchable)learning).getNeuralNet().save(getModelDir() + "/" + learning.getStepCount() + ".model");
            } catch (UnsupportedOperationException e) {
                String path = getModelDir() + "/" + learning.getStepCount();
                ((IActorCritic)((NeuralNetFetchable)learning).getNeuralNet()).save(path + "_value.model", path + "_policy.model");
            }
        }

    }

    @AllArgsConstructor
    @Value
    @Builder
    public static class Info {
        String trainingName;
        String mdpName;
        ILearningConfiguration conf;
        int stepCounter;
        long millisTime;
    }
}
