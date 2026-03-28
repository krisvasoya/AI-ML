// ════════════════════════════════════════════════════════════
// ML WORKER (kNN, Decision Tree, SVM, Logistic Regression)
// ════════════════════════════════════════════════════════════

let trainingData = [];
let testData = [];
let featureStats = {}; // min, max, median for 10 features
let classes = ['trivial', 'easy', 'medium', 'hard', 'very_hard'];

// The actual Models
let models = {
    knn: null,
    tree: null,
    svm: null,
    lr: null
};

// ── UTILITIES ──

function normalize(row) {
    const norm = {};
    for (const f in row) {
        if (!featureStats[f]) continue;
        const s = featureStats[f];
        if (s.max === s.min) norm[f] = 0;
        else norm[f] = (row[f] - s.min) / (s.max - s.min);
    }
    return norm;
}

function calculateFeatureStats(data) {
    if(!data || !data.length) return;
    const feats = Object.keys(data[0].features);
    featureStats = {};
    for(let f of feats) {
        let vals = data.map(d => d.features[f]).sort((a,b)=>a-b);
        featureStats[f] = {
            min: vals[0],
            max: vals[vals.length - 1],
            median: vals[Math.floor(vals.length / 2)]
        };
    }
}

function shuffle(array, seed) {
    let m = array.length, t, i;
    const random = () => {
        let x = Math.sin(seed++) * 10000;
        return x - Math.floor(x);
    };
    while (m) {
        i = Math.floor(random() * m--);
        t = array[m];
        array[m] = array[i];
        array[i] = t;
    }
    return array;
}

function computeMetrics(y_true, y_pred) {
    let correct = 0;
    const confusion = {};
    classes.forEach(c1 => {
        confusion[c1] = {};
        classes.forEach(c2 => confusion[c1][c2] = 0);
    });

    for (let i = 0; i < y_true.length; i++) {
        if (y_true[i] === y_pred[i]) correct++;
        confusion[y_true[i]][y_pred[i]]++;
    }

    const unweighted_precision = [];
    const unweighted_recall = [];

    classes.forEach(c => {
        let tp = confusion[c][c];
        let fp = 0, fn = 0;
        classes.forEach(other => {
            if (other !== c) {
                fp += confusion[other][c];
                fn += confusion[c][other];
            }
        });
        unweighted_precision.push( (tp + fp === 0) ? 0 : tp / (tp + fp) );
        unweighted_recall.push( (tp + fn === 0) ? 0 : tp / (tp + fn) );
    });

    const accuracy = correct / y_true.length;
    const macro_precision = unweighted_precision.reduce((a,b)=>a+b,0) / classes.length;
    const macro_recall = unweighted_recall.reduce((a,b)=>a+b,0) / classes.length;

    return { accuracy, macro_precision, macro_recall, confusion };
}

// ── 1. KNN CLASSIFIER ──
class KNN {
    constructor(k = 5) {
        this.k = k;
        this.savedData = [];
    }
    train(data) {
        this.savedData = data.map(d => ({
            features: normalize(d.features),
            label: d.label
        }));
    }
    predict(features) {
        const norm = normalize(features);
        let dists = this.savedData.map(d => {
            let distSq = 0;
            for (let f in d.features) {
                distSq += Math.pow(d.features[f] - norm[f], 2);
            }
            return { dist: distSq, label: d.label };
        });
        dists.sort((a,b) => a.dist - b.dist);
        let kNearest = dists.slice(0, this.k);
        let counts = {};
        kNearest.forEach(n => {
            counts[n.label] = (counts[n.label] || 0) + 1;
        });
        return Object.keys(counts).reduce((a,b) => counts[a] > counts[b] ? a : b);
    }
}

// ── 2. DECISION TREE (CART) ──
class DecisionTree {
    constructor(maxDepth) {
        this.maxDepth = maxDepth;
        this.tree = null;
        this.rules = [];
    }

    gini(data) {
        if (data.length === 0) return 0;
        let counts = {};
        data.forEach(d => counts[d.label] = (counts[d.label]||0) + 1);
        let impurity = 1;
        for (let k in counts) {
            let p = counts[k] / data.length;
            impurity -= p * p;
        }
        return impurity;
    }

    buildTree(data, depth) {
        let counts = {};
        data.forEach(d => counts[d.label] = (counts[d.label]||0) + 1);
        let majority = Object.keys(counts).reduce((a,b) => counts[a] > counts[b] ? a : b);

        if (depth >= this.maxDepth || data.length <= 1 || Object.keys(counts).length === 1) {
            return { prediction: majority };
        }

        let bestGini = Infinity, bestSplit = null;
        let featuresList = Object.keys(data[0].features);

        for (let f of featuresList) {
            let vals = [...new Set(data.map(d => d.features[f]))].sort((a,b)=>a-b);
            // evaluate thresholds
            for (let i = 0; i < vals.length - 1; i++) {
                let thresh = (vals[i] + vals[i+1]) / 2;
                let left = data.filter(d => d.features[f] <= thresh);
                let right = data.filter(d => d.features[f] > thresh);
                
                if(left.length === 0 || right.length === 0) continue;

                let g = (left.length/data.length)*this.gini(left) + (right.length/data.length)*this.gini(right);
                if (g < bestGini) {
                    bestGini = g;
                    bestSplit = { feature: f, threshold: thresh, leftData: left, rightData: right };
                }
            }
        }

        if (!bestSplit) return { prediction: majority };

        let node = {
            feature: bestSplit.feature,
            threshold: bestSplit.threshold,
            prediction: majority // fallback
        };
        
        node.left = this.buildTree(bestSplit.leftData, depth + 1);
        node.right = this.buildTree(bestSplit.rightData, depth + 1);
        return node;
    }

    train(data) {
        this.tree = this.buildTree(data, 0);
        this.extractRules(this.tree, 0, "");
    }

    extractRules(node, depth, prefix) {
        if (!node.left || depth >= 2) return;
        this.rules.push(`If ${node.feature} <= ${node.threshold.toFixed(2)} -> go left`);
        if(this.rules.length >= 3) return;
        this.extractRules(node.left, depth+1, prefix + "  ");
        this.extractRules(node.right, depth+1, prefix + "  ");
    }

    predict(features, node = this.tree) {
        if (!node.left && !node.right) return node.prediction;
        if (features[node.feature] <= node.threshold) return this.predict(features, node.left);
        return this.predict(features, node.right);
    }
}

// ── 3. SVM (Linear, OvR) ──
class LinearSVM {
    constructor() {
        this.models = {}; // binary classifier for each class
    }
    
    // Gradient descent for hinge loss
    trainBinary(data, targetClass) {
        let w = Object.keys(data[0].features).reduce((acc, k) => { acc[k] = 0; return acc; }, {});
        let b = 0;
        const lr = 0.05, lambda = 0.01;
        const epochs = 500;
        
        for (let ep = 0; ep < epochs; ep++) {
            data.forEach(d => {
                let y = (d.originalLabel === targetClass) ? 1 : -1;
                let dot = 0;
                for (let k in w) dot += w[k] * d.features[k];
                
                let condition = y * (dot + b) < 1;
                for (let k in w) {
                    if (condition) {
                        w[k] = w[k] - lr * (2 * lambda * w[k] - y * d.features[k]);
                    } else {
                        w[k] = w[k] - lr * (2 * lambda * w[k]);
                    }
                }
                if (condition) b = b + lr * y;
            });
        }
        return { w, b };
    }

    train(data) {
        // Normalise inputs internally
        let normData = data.map(d => ({ features: normalize(d.features), originalLabel: d.label }));
        classes.forEach(c => {
            this.models[c] = this.trainBinary(normData, c);
        });
    }

    predict(features) {
        let norm = normalize(features);
        let bestClass = null, maxScore = -Infinity;
        classes.forEach(c => {
            let m = this.models[c];
            let score = m.b;
            for (let f in m.w) score += m.w[f] * norm[f];
            if (score > maxScore) { maxScore = score; bestClass = c; }
        });
        return bestClass;
    }
}

// ── 4. LOGISTIC REGRESSION (Multinomial emulation via OvR) ──
class LogisticRegression {
    constructor() {
        this.models = {}; 
    }
    
    sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

    trainBinary(data, targetClass) {
        let w = Object.keys(data[0].features).reduce((acc, k) => { acc[k] = 0; return acc; }, {});
        let b = 0;
        const lr = 0.1;
        const epochs = 800; // log reg converges reasonably quickly on normalized data
        
        for (let ep = 0; ep < epochs; ep++) {
            let dw = {}, db = 0;
            for (let k in w) dw[k] = 0;
            
            data.forEach(d => {
                let y = (d.originalLabel === targetClass) ? 1 : 0;
                let z = b;
                for (let k in w) z += w[k] * d.features[k];
                let pred = this.sigmoid(z);
                let err = pred - y;
                
                db += err;
                for (let k in w) dw[k] += err * d.features[k];
            });
            
            b -= lr * db / data.length;
            for (let k in w) w[k] -= lr * dw[k] / data.length;
        }
        return { w, b };
    }

    train(data) {
        let normData = data.map(d => ({ features: normalize(d.features), originalLabel: d.label }));
        classes.forEach(c => {
            this.models[c] = this.trainBinary(normData, c);
        });
    }

    predict(features) {
        let norm = normalize(features);
        let bestClass = null, maxProb = -Infinity;
        classes.forEach(c => {
            let m = this.models[c];
            let z = m.b;
            for (let f in m.w) z += m.w[f] * norm[f];
            let prob = this.sigmoid(z);
            if (prob > maxProb) { maxProb = prob; bestClass = c; }
        });
        return bestClass;
    }
}


// ── WORKER EVENT HANDLER ──

self.onmessage = function(e) {
    const { type, data } = e.data;

    if (type === 'init') {
        const fullData = data;
        calculateFeatureStats(fullData);
        // Shuffle seeded so repeatable
        const shuffled = shuffle([...fullData], 42);
        const splitIdx = Math.floor(shuffled.length * 0.8);
        trainingData = shuffled.slice(0, splitIdx);
        testData = shuffled.slice(splitIdx);
        self.postMessage({ type: 'ready' });
    }
    
    if (type === 'train_all') {
        models.knn = new KNN();
        models.tree = new DecisionTree(6);
        models.svm = new LinearSVM();
        models.lr = new LogisticRegression();

        const results = [];

        // Train and Evaluate Helper
        const runModel = (id, label, modelInst) => {
            const t0 = performance.now();
            modelInst.train(trainingData);
            const trainTime = performance.now() - t0;

            const t1 = performance.now();
            const y_true = [];
            const y_pred = [];
            
            // test accuracy
            for (let d of testData) {
                y_true.push(d.label);
                y_pred.push(modelInst.predict(d.features));
            }
            
            // compute predict time over 50 random test points
            const tPred0 = performance.now();
            for(let i=0; i<50; i++){
                 modelInst.predict(testData[i%testData.length].features);
            }
            const avgPredTime = (performance.now() - tPred0) / 50;

            const metrics = computeMetrics(y_true, y_pred);
            
            let extra = {};
            if(id === 'tree') extra = { depth: 6, rules: modelInst.rules };
            
            results.push({
                id, label,
                metrics: {
                    accuracy: metrics.accuracy,
                    precision: metrics.macro_precision,
                    recall: metrics.macro_recall,
                    trainTime: trainTime,
                    predictTime: avgPredTime,
                    confusion: metrics.confusion
                },
                extra
            });
        };

        runModel('knn', 'k-Nearest Neighbor (k=5)', models.knn);
        runModel('tree', 'Decision Tree (CART)', models.tree);
        runModel('svm', 'Support Vector Machine (Linear)', models.svm);
        runModel('lr', 'Logistic Regression (Baseline)', models.lr);

        self.postMessage({ type: 'train_done', results });
    }

    if (type === 'predict') {
        const uFeat = data; // the user-provided features
        // Form full feature set
        const fullFeat = {};
        for(let f in featureStats) {
            fullFeat[f] = (uFeat[f] !== undefined) ? uFeat[f] : featureStats[f].median;
        }
        
        const preds = {};
        if (models.knn) preds.knn = models.knn.predict(fullFeat);
        if (models.tree) preds.tree = models.tree.predict(fullFeat);
        if (models.svm) preds.svm = models.svm.predict(fullFeat);
        if (models.lr) preds.lr = models.lr.predict(fullFeat);

        self.postMessage({ type: 'predict_done', predictions: preds, fullFeaturesUsed: fullFeat });
    }
};
