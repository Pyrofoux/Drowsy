const fs = require('fs');
let dirname = '../extracted';
let writename = '../formatted';

read();

//avatars();

let save = {};

save.sprites = [];

function read()
{
  fs.readdir(dirname, (err, files) =>
  {
    files.forEach(name =>
    {
      // Process only txt files
      if(name.split(".")[1] != "txt") return 0;

      let s = fs.readFileSync(dirname+'/'+name).toString();
      process(name,s);
    });
    write();
  });
}

function process(name,data)
{
  const blocks = data.split("\n\n");

  blocks.forEach(block =>
  {
    let label = identify(block);

    switch(label)
    {
      case "SPR":
        processSprite(block);
      break;

      case "TIL":
        processTile(name,block);
      break;

      case "ROOM":
        processRoom(name,block);
      break;

    }

  })

}


function identify(block)
{
  const label = block.split(/\s/)[0];
  return label;
}


function processSprite(block)
{
  const name = block.split(/\s/)[1];

  let data = block.split('\n');
    data.shift();
    data = data.splice(0,8);

    let bin = [];

    for(let i in data)
    {
      bin[i] = [];
      for(let j in data[i])
      {
        bin[i][j] = parseInt(data[i][j]);
      }
    }
    save.sprites.push(bin);
}

//Copy paste from just au dessus
function processTile(name, block)
{
  const name = block.split(/\s/)[1];

  let data = block.split('\n');
    data.shift();
    data = data.splice(0,8);

    let bin = [];

    for(let i in data)
    {
      bin[i] = [];
      for(let j in data[i])
      {
        bin[i][j] = parseInt(data[i][j]);
      }
    }
    save.sprites.push(bin);
}



function write()
{
	for(let i in save)
	{
	  fs.writeFileSync(writename+'/'+i+'.json',JSON.stringify(save[i]));
	  console.log("Wrote "+i+".json");
	}
}


// Format the ~400 avatars image
function avatars()
{
 var txt = fs.readFileSync(dirname+'/AVATARS.txt').toString();
 txt = txt.split("\n");

 var data = [];
 var lx = 20;
 var ly = 21;

 for(var gx = 0; gx < lx; gx++)
 {
   for(var gy = 0; gy < ly; gy++)
   {

     var spr = [];
     for(var x = 0; x < 8; x++)
     {
       spr[x] = [];
       for(var y = 0; y < 8; y++)
       {
        spr[x][y] = txt[gx*8+x][gy*8+y]*1;
       }
     }
     data.push(spr);
   }
   //console.log(data)
 }
data;
fs.writeFileSync(writename+'/avatars.json',JSON.stringify(data));
console.log("saved avatars")
}
