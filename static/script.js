// ── estado del modelo ────────────────────────────────────────────────────
async function checkModel(){
  try{
    const r=await fetch('/model_status');
    const d=await r.json();
    const pill=document.getElementById('modelPill');
    if(d.trained){pill.textContent='modelo listo';pill.className='status-pill ok'}
    else{pill.textContent='sin entrenar';pill.className='status-pill warn'}
  }catch(e){}
}
checkModel();

// ── detector ─────────────────────────────────────────────────────────────
const dropZone   =document.getElementById('dropZone');
const fileInput  =document.getElementById('fileInput');
const previewWrap=document.getElementById('previewWrap');
const previewImg =document.getElementById('previewImg');
const removeBtn  =document.getElementById('removeBtn');
const scanBtn    =document.getElementById('scanBtn');
const spinner    =document.getElementById('spinner');
const resultBox  =document.getElementById('resultBox');

let currentFile=null;

dropZone.addEventListener('click',()=>fileInput.click());
fileInput.addEventListener('change',()=>loadFile(fileInput.files[0]));
dropZone.addEventListener('dragover',e=>{e.preventDefault();dropZone.classList.add('over')});
dropZone.addEventListener('dragleave',()=>dropZone.classList.remove('over'));
dropZone.addEventListener('drop',e=>{
  e.preventDefault();dropZone.classList.remove('over');
  loadFile(e.dataTransfer.files[0]);
});
document.addEventListener('paste',e=>{
  const it=[...e.clipboardData.items].find(i=>i.type.startsWith('image/'));
  if(it)loadFile(it.getAsFile());
});

function loadFile(f){
  if(!f||!f.type.startsWith('image/'))return;
  currentFile=f;
  previewImg.src=URL.createObjectURL(f);
  dropZone.style.display='none';
  previewWrap.style.display='block';
  scanBtn.disabled=false;
  resultBox.style.display='none';
  spinner.style.display='none';
}

removeBtn.addEventListener('click',()=>{
  currentFile=null;fileInput.value='';
  previewWrap.style.display='none';
  dropZone.style.display='block';
  scanBtn.disabled=true;
  resultBox.style.display='none';
});

scanBtn.addEventListener('click',async()=>{
  if(!currentFile)return;
  scanBtn.disabled=true;
  spinner.style.display='block';
  resultBox.style.display='none';
  const fd=new FormData();fd.append('image',currentFile);
  try{
    const r=await fetch('/predict',{method:'POST',body:fd});
    const d=await r.json();
    if(!r.ok)throw new Error(d.error||'Error del servidor');
    showResult(d);
  }catch(e){
    alert('Error: '+e.message);
  }finally{
    spinner.style.display='none';
    scanBtn.disabled=false;
  }
});

function showResult(d){
  const esCon=d.label==='con_mascarilla';
  const verdict=document.getElementById('verdict');
  verdict.className='verdict '+(esCon?'con':'sin');
  document.getElementById('verdictEmoji').textContent=esCon?'😷':'🚫';
  document.getElementById('verdictText').textContent=esCon?'Con mascarilla':'Sin mascarilla';
  requestAnimationFrame(()=>setTimeout(()=>{
    document.getElementById('barCon').style.width=d.prob_con+'%';
    document.getElementById('barSin').style.width=d.prob_sin+'%';
  },60));
  document.getElementById('pctCon').textContent=d.prob_con+'%';
  document.getElementById('pctSin').textContent=d.prob_sin+'%';
  document.getElementById('metaFace').textContent=d.face_detected?'✓ cara detectada':'⚠ sin cara detectada';
  document.getElementById('metaConf').textContent='confianza: '+d.confidence+'%';
  resultBox.style.display='block';
}

// ── entrenamiento ─────────────────────────────────────────────────────────
const trainInput=document.getElementById('trainInput');
const trainFiles=document.getElementById('trainFiles');
const trainBtn  =document.getElementById('trainBtn');
const trainMsg  =document.getElementById('trainMsg');
let tFiles=[];

document.getElementById('pickTrainBtn').addEventListener('click',()=>trainInput.click());
trainInput.addEventListener('change',()=>addFiles([...trainInput.files]));

function addFiles(arr){
  arr.forEach(f=>{if(!tFiles.find(x=>x.name===f.name))tFiles.push(f)});
  renderFiles();
}
function renderFiles(){
  trainFiles.innerHTML='';
  tFiles.forEach((f,i)=>{
    const n=f.name.toLowerCase();
    let tag='unk',label='sin etiqueta';
    if(n.includes('con')){tag='con';label='con mascarilla'}
    else if(n.includes('sin')){tag='sin';label='sin mascarilla'}
    const row=document.createElement('div');
    row.className='file-row';
    row.innerHTML=`<span>${f.name}</span>
      <span class="tag tag-${tag}">${label}</span>
      <button onclick="delFile(${i})" style="background:none;border:none;cursor:pointer;color:var(--sub);font-size:12px">✕</button>`;
    trainFiles.appendChild(row);
  });
  trainBtn.disabled=tFiles.length<4;
}
window.delFile=i=>{tFiles.splice(i,1);renderFiles()};

trainBtn.addEventListener('click',async()=>{
  trainBtn.disabled=true;trainBtn.textContent='⏳ Entrenando…';
  showTrainMsg('Entrenando el modelo, esto puede tardar unos minutos…','info');
  const fd=new FormData();tFiles.forEach(f=>fd.append('images[]',f));
  try{
    const r=await fetch('/train',{method:'POST',body:fd});
    const d=await r.json();
    if(!r.ok)throw new Error(d.error||'Error');
    showTrainMsg(`✓ Listo · ${d.samples} muestras · precisión: ${d.accuracy}%`,'ok');
    checkModel();
  }catch(e){
    showTrainMsg('✗ '+e.message,'err');
  }finally{
    trainBtn.disabled=false;trainBtn.textContent='▶ Entrenar modelo';
  }
});

document.getElementById('driveBtn').addEventListener('click',async()=>{
  showTrainMsg('Descargando dataset desde Google Drive… (puede tardar varios minutos)','info');
  try{
    const r=await fetch('/train_from_drive',{method:'POST'});
    const d=await r.json();
    if(!r.ok)throw new Error(d.error||'Error');
    showTrainMsg(`✓ Descarga y entrenamiento completo · ${d.samples} muestras · precisión: ${d.accuracy}%`,'ok');
    checkModel();
  }catch(e){
    showTrainMsg('✗ '+e.message,'err');
  }
});

function showTrainMsg(txt,cls){
  trainMsg.textContent=txt;
  trainMsg.className=cls;
  trainMsg.style.display='block';
}
