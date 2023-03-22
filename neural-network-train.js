import * as tf from '@tensorflow/tfjs';

// Dataset
const dataset = [
    { name: 'Electron', spin: 0.5, charge: -1, flavor: 0, color: 0, mass: 0.511 },
    { name: 'Electron Neutrino', spin: 0.5, charge: 0, flavor: 0, color: 0, mass: 1e-9 },
    { name: 'Muon', spin: 0.5, charge: -1, flavor: 0, color: 0, mass: 105.7 },
    { name: 'Muon Neutrino', spin: 0.5, charge: 0, flavor: 0, color: 0, mass: 0.17 },
    { name: 'Tau', spin: 0.5, charge: -1, flavor: 0, color: 0, mass: 1776.8 },
    { name: 'Tau Neutrino', spin: 0.5, charge: 0, flavor: 0, color: 0, mass: 15 },
    { name: 'Up Quark', spin: 0.5, charge: 2 / 3, flavor: 1, color: 1, mass: 2.3 },
    { name: 'Down Quark', spin: 0.5, charge: -1 / 3, flavor: 1, color: 1, mass: 4.8 },
    { name: 'Charm Quark', spin: 0.5, charge: 2 / 3, flavor: 1, color: 1, mass: 1275 },
    { name: 'Strange Quark', spin: 0.5, charge: -1 / 3, flavor: 1, color: 1, mass: 95 },
    { name: 'Top Quark', spin: 0.5, charge: 2 / 3, flavor: 1, color: 1, mass: 173000 },
    { name: 'Bottom Quark', spin: 0.5, charge: -1 / 3, flavor: 1, color: 1, mass: 4180 },
    { name: 'Gluon', spin: 1, charge: 0, flavor: 2, color: 1, mass: 0 },
    { name: 'Photon', spin: 1, charge: 0, flavor: 2, color: 0, mass: 0 },
    { name: 'Z Boson', spin: 1, charge: 0, flavor: 2, color: 0, mass: 91187.6 },
    { name: 'W+ Boson', spin: 1, charge: 1, flavor: 2, color: 0, mass: 80379 },
    { name: 'W- Boson', spin: 1, charge: -1, flavor: 2, color: 0, mass: 80379 },
];


// Preprocessing
const featureScaler = data => {
    // Apply any scaling or normalization here
    return data;
};

const massScaler = mass => {
    // Scale the mass if needed, e.g. by dividing by a constant
    return mass;
};

const trainData = dataset.map(data => {
    return {
        x: featureScaler([data.spin, data.charge, data.flavor, data.color]),
        y: massScaler(data.mass),
    };
});

// Model
const model = tf.sequential();
model.add(tf.layers.dense({ units: 32, inputShape: [4], activation: 'relu' }));
model.add(tf.layers.dense({
    units: 16, activation: 'relu'
}));
model.add(tf.layers.dense({ units: 1, activation: 'linear' }));

// Compile the model
model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse'],
});

// Train the model
const xs = tf.tensor2d(trainData.map(d => d.x));
const ys = tf.tensor2d(trainData.map(d => d.y), [trainData.length, 1]);

(async function () {
    const history = await model.fit(xs, ys, {
        batchSize: 4,
        epochs: 100,
        shuffle: true,
    });

    console.log('Training complete.');
})();

// Predict
const predict = (spin, charge, flavor, color) => {
    const input = tf.tensor2d(featureScaler([spin, charge, flavor, color]), [1, 4]);
    const output = model.predict(input);
    const mass = massScaler(output.dataSync()[0]); // Inverse the mass scaling here if needed
    return mass;
};

// Example prediction
console.log(predict(0.5, -1, 0, 0)); // Should output a value close to 0.511 (Electron mass)
