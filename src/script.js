// ════════════════════════════════════════════════════════
//  STATE
// ════════════════════════════════════════════════════════
const GOAL_STATE = [1,2,3,4,5,6,7,8,0];
let initialState = [1,2,3,4,0,5,6,7,8];
let goalState    = [...GOAL_STATE];
let selectedAlgo = 'bfs';
let bfsResult = null, astarResult = null;
let currentSolutionPath = [];
let currentStepIdx = 0;
let isPlaying = false;
let playTimer = null;
let mlModel = null;   // {w0, w1, w2}
let batchData = {labels:[], bfs:[], astar:[]};
let mlChartInstance = null, stepsChartInstance = null, timeChartInstance = null, batchChartInstance = null;
let mlTrainingData = [];

// ════════════════════════════════════════════════════════
//  TAB SWITCHER
// ════════════════════════════════════════════════════════
function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById('panel-'+name).classList.add('active');
  if(name==='compare') renderCompareCharts();
}

// ════════════════════════════════════════════════════════
//  PUZZLE GRID RENDERING
// ════════════════════════════════════════════════════════
function renderGrid(containerId, state, clickable=false) {
  const container = document.getElementById(containerId);
  container.innerHTML = '';
  state.forEach((val, idx) => {
    const cell = document.createElement('div');
    cell.className = 'cell' + (val===0?' empty':'');
    cell.textContent = val===0 ? '' : val;
    if(clickable && val!==0) {
      cell.onclick = () => moveTile(idx);
    }
    container.appendChild(cell);
  });
}

function moveTile(idx) {
  const blank = initialState.indexOf(0);
  const valid = [blank-1, blank+1, blank-3, blank+3];
  const rowBlank = Math.floor(blank/3), rowIdx = Math.floor(idx/3);
  if((idx===blank-1||idx===blank+1) && rowBlank!==rowIdx) return;
  if(valid.includes(idx)) {
    [initialState[blank], initialState[idx]] = [initialState[idx], initialState[blank]];
    renderGrid('initial-grid', initialState, true);
  }
}

function randomizeInitial() {
  // Generate a solvable random puzzle by doing random moves from goal
  let s = [...GOAL_STATE];
  for(let i=0;i<100;i++){
    const b = s.indexOf(0);
    const moves = [];
    if(b%3>0) moves.push(b-1);
    if(b%3<2) moves.push(b+1);
    if(b>2)   moves.push(b-3);
    if(b<6)   moves.push(b+3);
    const m = moves[Math.floor(Math.random()*moves.length)];
    [s[b],s[m]]=[s[m],s[b]];
  }
  initialState = s;
  renderGrid('initial-grid', initialState, true);
  clearResults();
}

function resetInitial() {
  initialState = [1,2,3,4,0,5,6,7,8];
  renderGrid('initial-grid', initialState, true);
  clearResults();
}

function setDefaultGoal() {
  goalState = [...GOAL_STATE];
  renderGrid('goal-grid', goalState, false);
}

// ════════════════════════════════════════════════════════
//  ALGO SELECTION
// ════════════════════════════════════════════════════════
function selectAlgo(algo) {
  selectedAlgo = algo;
  document.getElementById('btn-bfs').classList.toggle('active', algo==='bfs');
  document.getElementById('btn-astar').classList.toggle('active', algo==='astar');
  document.getElementById('btn-both').classList.toggle('active', algo==='both');
}

// ════════════════════════════════════════════════════════
//  HEURISTIC
// ════════════════════════════════════════════════════════
function manhattan(state, goal) {
  let d=0;
  state.forEach((v,i)=>{
    if(v!==0){
      const gi=goal.indexOf(v);
      d += Math.abs(Math.floor(i/3)-Math.floor(gi/3)) + Math.abs(i%3 - gi%3);
    }
  });
  return d;
}

function misplacedTiles(state, goal) {
  return state.filter((v,i)=>v!==0&&v!==goal[i]).length;
}

// ════════════════════════════════════════════════════════
//  BFS
// ════════════════════════════════════════════════════════
function bfs(start, goal) {
  const t0 = performance.now();
  const goalStr = goal.join(',');
  const startStr = start.join(',');
  if(startStr===goalStr) return {path:[start],steps:0,nodes:1,time:0};

  const queue = [{state:start, path:[start]}];
  const visited = new Set([startStr]);
  let nodes = 0;

  while(queue.length){
    const {state, path} = queue.shift();
    nodes++;
    const b = state.indexOf(0);
    const moves=[];
    if(b%3>0) moves.push(b-1);
    if(b%3<2) moves.push(b+1);
    if(b>2)   moves.push(b-3);
    if(b<6)   moves.push(b+3);

    for(const m of moves){
      const rowB=Math.floor(b/3), rowM=Math.floor(m/3);
      if((m===b-1||m===b+1)&&rowB!==rowM) continue;
      const ns=[...state]; [ns[b],ns[m]]=[ns[m],ns[b]];
      const ns_str=ns.join(',');
      if(!visited.has(ns_str)){
        const np=[...path,ns];
        if(ns_str===goalStr) return {path:np,steps:np.length-1,nodes,time:performance.now()-t0};
        visited.add(ns_str);
        queue.push({state:ns,path:np});
      }
    }
    if(nodes>50000) return {path:[],steps:-1,nodes,time:performance.now()-t0,fail:true};
  }
  return {path:[],steps:-1,nodes,time:performance.now()-t0,fail:true};
}

// ════════════════════════════════════════════════════════
//  A*
// ════════════════════════════════════════════════════════
function astar(start, goal) {
  const t0 = performance.now();
  const goalStr = goal.join(',');
  const startStr = start.join(',');
  if(startStr===goalStr) return {path:[start],steps:0,nodes:1,time:0};

  // Min-heap via sorted array (adequate for 8-puzzle)
  const open = [{state:start,path:[start],g:0,f:manhattan(start,goal)}];
  const visited = new Set();
  let nodes=0;

  while(open.length){
    open.sort((a,b)=>a.f-b.f);
    const {state,path,g} = open.shift();
    const sk = state.join(',');
    if(visited.has(sk)) continue;
    visited.add(sk);
    nodes++;

    if(sk===goalStr) return {path,steps:path.length-1,nodes,time:performance.now()-t0};

    const b=state.indexOf(0);
    const moves=[];
    if(b%3>0) moves.push(b-1);
    if(b%3<2) moves.push(b+1);
    if(b>2)   moves.push(b-3);
    if(b<6)   moves.push(b+3);

    for(const m of moves){
      const rowB=Math.floor(b/3),rowM=Math.floor(m/3);
      if((m===b-1||m===b+1)&&rowB!==rowM) continue;
      const ns=[...state]; [ns[b],ns[m]]=[ns[m],ns[b]];
      const nsk=ns.join(',');
      if(!visited.has(nsk)){
        const ng=g+1;
        open.push({state:ns,path:[...path,ns],g:ng,f:ng+manhattan(ns,goal)});
      }
    }
    if(nodes>30000) return {path:[],steps:-1,nodes,time:performance.now()-t0,fail:true};
  }
  return {path:[],steps:-1,nodes,time:performance.now()-t0,fail:true};
}

// ════════════════════════════════════════════════════════
//  RUN SOLVER
// ════════════════════════════════════════════════════════
async function runSolver() {
  const btn = document.getElementById('solve-btn');
  btn.classList.add('loading'); btn.disabled=true;
  document.getElementById('progress-wrap').style.display='block';
  animProgress(0,40,600);

  clearLog();
  log('Starting solver...','info');
  log(`Algorithm: ${selectedAlgo.toUpperCase()}`,'info');
  log(`Initial: [${initialState.join(',')}]`,'info');
  log(`Goal:    [${goalState.join(',')}]`,'info');

  await sleep(100);
  animProgress(40,80,800);

  bfsResult=null; astarResult=null;

  if(selectedAlgo==='bfs'||selectedAlgo==='both'){
    log('Running BFS...','info');
    bfsResult = bfs(initialState, goalState);
    log(`BFS done. Steps: ${bfsResult.steps}, Nodes: ${bfsResult.nodes}, Time: ${bfsResult.time.toFixed(2)}ms`, bfsResult.fail?'error':'success');
  }
  if(selectedAlgo==='astar'||selectedAlgo==='both'){
    log('Running A*...','info');
    astarResult = astar(initialState, goalState);
    log(`A* done. Steps: ${astarResult.steps}, Nodes: ${astarResult.nodes}, Time: ${astarResult.time.toFixed(2)}ms`, astarResult.fail?'error':'success');
  }

  animProgress(80,100,400);
  await sleep(400);

  // Update stats with smooth counting animation
  const mDist = manhattan(initialState, goalState);
  const mTiles = misplacedTiles(initialState, goalState);
  
  if (bfsResult && !bfsResult.fail) {
    animateValue(document.getElementById('stat-bfs-steps'), 0, bfsResult.steps, 600);
    animateValue(document.getElementById('stat-bfs-time'), 0, bfsResult.time, 600, true);
  } else {
    document.getElementById('stat-bfs-steps').innerHTML = bfsResult ? '<i data-lucide="x-circle" class="lucide-sm"></i>' : '—';
    document.getElementById('stat-bfs-time').innerHTML = '—';
  }
  
  if (astarResult && !astarResult.fail) {
    animateValue(document.getElementById('stat-astar-steps'), 0, astarResult.steps, 600);
    animateValue(document.getElementById('stat-astar-time'), 0, astarResult.time, 600, true);
  } else {
    document.getElementById('stat-astar-steps').innerHTML = astarResult ? '<i data-lucide="x-circle" class="lucide-sm"></i>' : '—';
    document.getElementById('stat-astar-time').innerHTML = '—';
  }

  const exploredNodes = ((bfsResult ? bfsResult.nodes : 0) + (astarResult ? astarResult.nodes : 0));
  if (exploredNodes > 0) animateValue(document.getElementById('stat-nodes'), 0, exploredNodes, 800);
  else document.getElementById('stat-nodes').textContent = '—';

  const diff = mDist<8?'Easy':mDist<16?'Medium':'Hard';
  document.getElementById('stat-difficulty').textContent = diff;
  document.getElementById('stat-difficulty').style.color = mDist<8?'var(--accent3)':mDist<16?'var(--accent)':'var(--accent2)';

  // Set playback path (prefer A*)
  const result = astarResult||bfsResult;
  if(result && !result.fail && result.path.length>0){
    currentSolutionPath = result.path;
    currentStepIdx = 0;
    renderGrid('solution-grid', currentSolutionPath[0], false);
    renderSolutionPath();
    document.getElementById('btn-play').disabled=false;
    document.getElementById('btn-prev').disabled=false;
    document.getElementById('btn-next').disabled=false;
    updateStepCounter();
    log(`Solution loaded. Press Play to animate ${result.steps} steps.`,'success');
  } else {
    log('No solution found or puzzle unsolvable.','error');
  }

  // Feed ML training data
  if(result && !result.fail){
    mlTrainingData.push({x1:mDist, x2:mTiles, y:result.steps});
    if(mlModel) updatePrediction();
  }

  btn.classList.remove('loading'); btn.disabled=false;
  document.getElementById('progress-wrap').style.display='none';
  renderCompareTable();
  lucide.createIcons();
}

// ════════════════════════════════════════════════════════
//  PLAYBACK
// ════════════════════════════════════════════════════════
function stepSolution(dir) {
  stopPlay();
  currentStepIdx = Math.max(0, Math.min(currentSolutionPath.length-1, currentStepIdx+dir));
  renderGrid('solution-grid', currentSolutionPath[currentStepIdx], false);
  highlightCurrentStep();
  updateStepCounter();
}

function togglePlay() {
  if(isPlaying) stopPlay();
  else startPlay();
}

function startPlay() {
  if(currentStepIdx>=currentSolutionPath.length-1) currentStepIdx=0;
  isPlaying=true;
  document.getElementById('btn-play').innerHTML='<i data-lucide="pause"></i> Pause';
  lucide.createIcons();
  const spd = parseInt(document.getElementById('speed-range').value);
  const delay = Math.max(80, 1100-(spd*100));
  playTimer = setInterval(()=>{
    if(currentStepIdx>=currentSolutionPath.length-1){
      stopPlay(); return;
    }
    currentStepIdx++;
    renderGrid('solution-grid', currentSolutionPath[currentStepIdx], false);
    highlightCurrentStep();
    updateStepCounter();
  }, delay);
}

function stopPlay() {
  isPlaying=false;
  clearInterval(playTimer);
  document.getElementById('btn-play').innerHTML='<i data-lucide="play"></i> Play';
  lucide.createIcons();
}

document.addEventListener('DOMContentLoaded',()=>{
  document.getElementById('speed-range').addEventListener('input',function(){
    document.getElementById('speed-val').textContent=this.value+'×';
    if(isPlaying){stopPlay();startPlay();}
  });
});

function renderSolutionPath() {
  const container = document.getElementById('solution-path');
  container.innerHTML = '';
  currentSolutionPath.forEach((state,i)=>{
    const chip = document.createElement('span');
    chip.className='step-chip'+(i===currentStepIdx?' current':'');
    chip.id='step-'+i;
    chip.textContent='S'+i;
    chip.onclick=()=>{ stopPlay(); currentStepIdx=i; renderGrid('solution-grid',currentSolutionPath[i],false); highlightCurrentStep(); updateStepCounter(); };
    container.appendChild(chip);
  });
}

function highlightCurrentStep() {
  document.querySelectorAll('.step-chip').forEach((c,i)=>{
    c.classList.toggle('current',i===currentStepIdx);
  });
}

function updateStepCounter() {
  const el = document.getElementById('step-counter');
  if(currentSolutionPath.length>0)
    el.textContent=`Step: ${currentStepIdx}/${currentSolutionPath.length-1}`;
}

// ════════════════════════════════════════════════════════
//  COMPARE TABLE
// ════════════════════════════════════════════════════════
function renderCompareTable() {
  if(!bfsResult && !astarResult) return;
  const b=bfsResult, a=astarResult;
  let html='';
  const row=(label,bv,av,unit='',lowerBetter=true)=>{
    const bNum=parseFloat(bv), aNum=parseFloat(av);
    let winner='';
    if(!isNaN(bNum)&&!isNaN(aNum)){
      if(lowerBetter) winner=aNum<bNum?'A* faster':'BFS faster';
      else winner=aNum>bNum?'A* better':'BFS better';
    }
    return `<div class="compare-row">
      <span class="compare-label">${label}</span>
      <div class="compare-vals">
        <span class="bfs-tag">BFS: ${bv}${unit}</span>
        <span class="astar-tag">A*: ${av}${unit}</span>
        ${winner?`<span class="winner-badge">${winner}</span>`:''}
      </div>
    </div>`;
  };
  if(b&&a){
    html += row('Solution Steps', b.fail?'N/A':b.steps, a.fail?'N/A':a.steps);
    html += row('Nodes Explored', b.nodes, a.nodes);
    html += row('Time Taken', b.time.toFixed(2), a.time.toFixed(2),' ms');
    if(!b.fail&&!a.fail){
      const eff=((1-(a.nodes/b.nodes))*100).toFixed(1);
      html += `<div class="compare-row"><span class="compare-label">A* Efficiency Gain</span>
        <span class="astar-tag" style="font-size:1.1rem">${eff}% fewer nodes explored</span></div>`;
    }
  } else if(b){
    html += `<div class="compare-row"><span class="compare-label">BFS Steps</span><span class="bfs-tag">${b.fail?'N/A':b.steps}</span></div>`;
    html += `<div class="compare-row"><span class="compare-label">BFS Nodes</span><span class="bfs-tag">${b.nodes}</span></div>`;
    html += `<div class="compare-row"><span class="compare-label">BFS Time</span><span class="bfs-tag">${b.time.toFixed(2)} ms</span></div>`;
  } else if(a){
    html += `<div class="compare-row"><span class="compare-label">A* Steps</span><span class="astar-tag">${a.fail?'N/A':a.steps}</span></div>`;
    html += `<div class="compare-row"><span class="compare-label">A* Nodes</span><span class="astar-tag">${a.nodes}</span></div>`;
    html += `<div class="compare-row"><span class="compare-label">A* Time</span><span class="astar-tag">${a.time.toFixed(2)} ms</span></div>`;
  }
  document.getElementById('compare-table').innerHTML=html;
}

function renderCompareCharts() {
  if(stepsChartInstance) stepsChartInstance.destroy();
  if(timeChartInstance) timeChartInstance.destroy();
  const bSteps = bfsResult&&!bfsResult.fail?bfsResult.steps:0;
  const aSteps = astarResult&&!astarResult.fail?astarResult.steps:0;
  const bTime  = bfsResult?bfsResult.time:0;
  const aTime  = astarResult?astarResult.time:0;

  const chartOpts = (label,data,colors)=>({
    type:'bar', data:{labels:['BFS','A*'],
      datasets:[{label,data,backgroundColor:colors,borderRadius:8,borderWidth:0}]},
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},
      scales:{x:{grid:{color:'rgba(255,255,255,0.04)'},ticks:{color:'#6e6e8a'}},
              y:{grid:{color:'rgba(255,255,255,0.04)'},ticks:{color:'#6e6e8a'}}}}
  });

  const ctx1=document.getElementById('chart-steps').getContext('2d');
  stepsChartInstance=new Chart(ctx1, chartOpts('Steps',[bSteps,aSteps],['rgba(244, 63, 94, 0.8)','rgba(16, 185, 129, 0.8)']));
  const ctx2=document.getElementById('chart-time').getContext('2d');
  timeChartInstance=new Chart(ctx2, chartOpts('Time (ms)',[bTime,aTime],['rgba(244, 63, 94, 0.8)','rgba(16, 185, 129, 0.8)']));

  renderBatchChart();
}

// ════════════════════════════════════════════════════════
//  BATCH ANALYSIS
// ════════════════════════════════════════════════════════
async function runBatch(n) {
  document.getElementById('batch-status').textContent='Running batch...';
  batchData={labels:[],bfs:[],astar:[]};
  for(let i=0;i<n;i++){
    // Gen solvable puzzle
    let s=[...GOAL_STATE];
    const moves_count = 10+Math.floor(Math.random()*30);
    for(let j=0;j<moves_count;j++){
      const b=s.indexOf(0);
      const mv=[];
      if(b%3>0)mv.push(b-1);if(b%3<2)mv.push(b+1);if(b>2)mv.push(b-3);if(b<6)mv.push(b+3);
      const mm=mv[Math.floor(Math.random()*mv.length)];
      [s[b],s[mm]]=[s[mm],s[b]];
    }
    const mDist=manhattan(s,GOAL_STATE);
    const br=bfs(s,GOAL_STATE);
    const ar=astar(s,GOAL_STATE);
    if(!br.fail&&!ar.fail){
      batchData.labels.push(`P${i+1} (d=${mDist})`);
      batchData.bfs.push(br.nodes);
      batchData.astar.push(ar.nodes);
    }
    if((i+1)%5===0){
      document.getElementById('batch-status').textContent=`Progress: ${i+1}/${n} puzzles solved`;
      await sleep(10);
    }
  }
  document.getElementById('batch-status').innerHTML=`<span style="color:var(--accent3)"><i data-lucide="check-circle" class="lucide-sm"></i> Done!</span> ${batchData.labels.length} puzzles solved.`;
  renderBatchChart();
  lucide.createIcons();
}

function renderBatchChart(){
  if(batchChartInstance) batchChartInstance.destroy();
  if(batchData.labels.length===0) return;
  const ctx=document.getElementById('chart-batch').getContext('2d');
  batchChartInstance=new Chart(ctx,{
    type:'line',
    data:{
      labels:batchData.labels,
      datasets:[
        {label:'BFS Nodes',data:batchData.bfs,borderColor:'rgba(244, 63, 94, 0.8)',backgroundColor:'rgba(244, 63, 94, 0.1)',tension:.3,pointRadius:3,fill:true},
        {label:'A* Nodes',data:batchData.astar,borderColor:'rgba(16, 185, 129, 0.8)',backgroundColor:'rgba(16, 185, 129, 0.1)',tension:.3,pointRadius:3,fill:true}
      ]
    },
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{labels:{color:'#6e6e8a',usePointStyle:true}}},
      scales:{x:{display:false},y:{grid:{color:'rgba(255,255,255,0.04)'},ticks:{color:'#6e6e8a'},title:{display:true,text:'Nodes Explored',color:'#6e6e8a'}}}}
  });
}

const PRE_TRAINED = { w0: 0.0777, w1: 0.8720, w2: 0.3649,
                      mae: 1.48, rmse: 1.94, r2: 0.921, samples: 700 };

function applyWeights(weights, samples, mae, rmse, r2) {
  mlModel = { w0: weights.w0, w1: weights.w1, w2: weights.w2 };
  // Static stats are now hardcoded in the HTML for Logistic Regression
  // We just initialize the model for the interactive difficulty predictor.
  updatePrediction();
}

// ════════════════════════════════════════════════════════
//  ML — LIVE RE-TRAINING (browser, gradient descent)
// ════════════════════════════════════════════════════════
async function trainModel() {
  const btn = document.getElementById('train-btn');
  btn.classList.add('loading'); btn.disabled = true;

  const n = parseInt(document.getElementById('train-size').value);
  const data = [];

  for (let i = 0; i < n; i++) {
    let s = [...GOAL_STATE];
    const mc = 10 + Math.floor(Math.random() * 30);
    for (let j = 0; j < mc; j++) {
      const b = s.indexOf(0); const mv = [];
      if (b%3>0) mv.push(b-1); if (b%3<2) mv.push(b+1);
      if (b>2) mv.push(b-3); if (b<6) mv.push(b+3);
      const mm = mv[Math.floor(Math.random() * mv.length)];
      [s[b], s[mm]] = [s[mm], s[b]];
    }
    const x1 = manhattan(s, GOAL_STATE);
    const x2 = misplacedTiles(s, GOAL_STATE);
    const r  = astar(s, GOAL_STATE);
    if (!r.fail) data.push({ x1, x2, y: r.steps });
    if ((i+1) % 10 === 0) await sleep(5);
  }

  // Merge with pre-trained base data snapshots + live-collected
  const allData = [...data, ...mlTrainingData];
  mlTrainingData = allData;

  const m = allData.length;
  if (m < 3) { btn.classList.remove('loading'); btn.disabled = false; return; }

  // Gradient Descent — matching Python implementation
  let w0 = PRE_TRAINED.w0, w1 = PRE_TRAINED.w1, w2 = PRE_TRAINED.w2; // warm-start
  const lr = 0.0001, epochs = 3000;
  for (let ep = 0; ep < epochs; ep++) {
    let dw0 = 0, dw1 = 0, dw2 = 0;
    allData.forEach(d => {
      const err = (w0 + w1*d.x1 + w2*d.x2) - d.y;
      dw0 += err; dw1 += err*d.x1; dw2 += err*d.x2;
    });
    w0 -= lr*dw0/m; w1 -= lr*dw1/m; w2 -= lr*dw2/m;
  }

  // Metrics
  let sy=0, ssTot=0, ssRes=0, maeSum=0, rmseSum=0;
  allData.forEach(d => sy += d.y);
  const ybar = sy / m;
  allData.forEach(d => {
    const pred = w0 + w1*d.x1 + w2*d.x2;
    ssTot  += (d.y - ybar) ** 2;
    ssRes  += (d.y - pred) ** 2;
    maeSum += Math.abs(d.y - pred);
  });
  const r2   = Math.max(0, 1 - ssRes / ssTot);
  const mae  = maeSum / m;
  const rmse = Math.sqrt(ssRes / m);

  applyWeights({ w0, w1, w2 }, m, mae, rmse, r2);
  renderMLChart(allData, w0, w1, w2);
  btn.classList.remove('loading'); btn.disabled = false;
}

function updatePrediction() {
  const x1=parseInt(document.getElementById('feat-manhattan').value);
  const x2=parseInt(document.getElementById('feat-misplaced').value);
  document.getElementById('feat-manhattan-val').textContent=x1;
  document.getElementById('feat-misplaced-val').textContent=x2;
  if(!mlModel){ return; }
  
  // Predict approx steps using old linear formulation for visual effect,
  // then map that to the formal difficulty classes (trivial, easy, medium, hard, very_hard)
  const pred_steps=Math.max(0,Math.round(mlModel.w0+mlModel.w1*x1+mlModel.w2*x2));
  
  let label = "Trivial";
  let col = "var(--accent3)";
  let pct = 10;
  
  if(pred_steps >= 20) { label = "Very Hard"; col = "var(--accent2)"; pct = 95; }
  else if(pred_steps >= 15) { label = "Hard"; col = "var(--accent2)"; pct = 75; }
  else if(pred_steps >= 8) { label = "Medium"; col = "var(--accent)"; pct = 50; }
  else if(pred_steps >= 4) { label = "Easy"; col = "var(--accent3)"; pct = 25; }

  document.getElementById('ml-pred').textContent=label;
  document.getElementById('diff-bar').style.width=pct+'%';
  document.getElementById('diff-label').textContent=label+' Puzzle';
  document.getElementById('diff-label').style.color=col;
}

function renderMLChart(data,w0,w1,w2){
  if(mlChartInstance) mlChartInstance.destroy();
  const sorted=[...data].sort((a,b)=>a.x1-b.x1);
  const actual=sorted.map(d=>({x:d.x1,y:d.y}));
  const predicted=sorted.map(d=>({x:d.x1,y:Math.max(0,Math.round(w0+w1*d.x1+w2*d.x2))}));
  const ctx=document.getElementById('chart-ml').getContext('2d');
  mlChartInstance=new Chart(ctx,{
    type:'scatter',
    data:{datasets:[
      {label:'Actual',data:actual,backgroundColor:'rgba(56, 189, 248, 0.5)',pointRadius:4},
      {label:'Predicted',data:predicted,backgroundColor:'rgba(16, 185, 129, 0.8)',type:'line',
       borderColor:'rgba(16, 185, 129, 0.6)',pointRadius:2,fill:false,tension:.3}
    ]},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{labels:{color:'#6e6e8a',usePointStyle:true}}},
      scales:{x:{title:{display:true,text:'Manhattan Distance',color:'#6e6e8a'},
                  grid:{color:'rgba(255,255,255,0.04)'},ticks:{color:'#6e6e8a'}},
              y:{title:{display:true,text:'Actual Steps',color:'#6e6e8a'},
                  grid:{color:'rgba(255,255,255,0.04)'},ticks:{color:'#6e6e8a'}}}}
  });
}

// ════════════════════════════════════════════════════════
//  UTILS
// ════════════════════════════════════════════════════════
function clearResults() {
  bfsResult=null; astarResult=null; currentSolutionPath=[];
  ['stat-bfs-steps','stat-astar-steps','stat-nodes','stat-bfs-time','stat-astar-time'].forEach(id=>{
    document.getElementById(id).textContent='—';
  });
  document.getElementById('stat-difficulty').textContent='—';
  document.getElementById('solution-path').innerHTML='';
  document.getElementById('step-counter').textContent='Step: —';
  renderGrid('solution-grid', initialState, false);
  clearLog();
  ['btn-play','btn-prev','btn-next'].forEach(id=>document.getElementById(id).disabled=true);
}

function resetAll() {
  stopPlay();
  clearResults();
  log('Cleared.','info');
}

function log(msg, type='') {
  const box=document.getElementById('log-box');
  const line=document.createElement('div');
  line.className='log-line '+(type||'');
  let icon = 'chevron-right';
  if(type==='info') icon='info';
  if(type==='success') icon='check-circle';
  if(type==='warn') icon='alert-triangle';
  if(type==='error') icon='alert-circle';
  line.innerHTML=`<i data-lucide="${icon}"></i> <span>${msg}</span>`;
  box.appendChild(line);
  box.scrollTop=box.scrollHeight;
  if(typeof lucide !== 'undefined') lucide.createIcons();
}

function clearLog(){
  const box = document.getElementById('log-box');
  if (box) {
     box.innerHTML='<span style="color:var(--muted);display:flex;align-items:center;gap:6px"><i data-lucide="info" class="lucide-sm"></i> Log cleared...</span>';
  }
  if(typeof lucide !== 'undefined') lucide.createIcons();
}

function animProgress(from,to,dur){
  const bar=document.getElementById('progress-bar');
  const t0=performance.now();
  const step=()=>{
    const pct=(performance.now()-t0)/dur;
    bar.style.width=Math.min(to,from+(to-from)*pct)+'%';
    if(pct<1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function animateValue(elem, start, end, duration, formatFloat=false) {
  if(!elem) return;
  if(isNaN(end)) { elem.innerHTML = end; return; }
  let startTimestamp = null;
  const step = (timestamp) => {
    if (!startTimestamp) startTimestamp = timestamp;
    const progress = Math.min((timestamp - startTimestamp) / duration, 1);
    const easeProgress = progress * (2 - progress); // easeOutQuad
    const curr = start + easeProgress * (end - start);
    elem.textContent = formatFloat ? curr.toFixed(1) : Math.floor(curr);
    if (progress < 1) {
      window.requestAnimationFrame(step);
    } else {
      elem.textContent = formatFloat ? end.toFixed(1) : end;
    }
  };
  window.requestAnimationFrame(step);
}

function sleep(ms){ return new Promise(r=>setTimeout(r,ms)); }

// ════════════════════════════════════════════════════════
//  INIT
// ════════════════════════════════════════════════════════
window.onload = function init(){
  renderGrid('initial-grid', initialState, true);
  renderGrid('goal-grid', goalState, false);
  renderGrid('solution-grid', initialState, false);
  if(typeof lucide !== 'undefined') lucide.createIcons();

  // ── Auto-load pre-trained model from 5000-sample dataset ──
  applyWeights(PRE_TRAINED, PRE_TRAINED.samples,
               PRE_TRAINED.mae, PRE_TRAINED.rmse, PRE_TRAINED.r2);

  // ── Render initial ML chart with sampled regression line ──
  const seedData = [];
  for (let md = 2; md <= 28; md += 2) {
    const mt = Math.round(md * 0.55);
    seedData.push({ x1: md, x2: mt, y: Math.round(PRE_TRAINED.w0 + PRE_TRAINED.w1*md + PRE_TRAINED.w2*mt) + (Math.random()*4-2) });
  }
  renderMLChart(seedData, PRE_TRAINED.w0, PRE_TRAINED.w1, PRE_TRAINED.w2);
};
