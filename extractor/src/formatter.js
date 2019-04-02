const fs = require('fs');
let dirname = '../extracted';
let writename = '../formatted';

read();
//avatars();

let save = {};

save.sprites = [];
save.tiles = {};
save.roomCompressed = [];
save.rooms = [];

function read()
{
  fs.readdir(dirname, (err, files) =>
  {
    console.log(__filename+" <<<")
    files.forEach(name =>
    {
      // Process only txt files
      if(name.split(".")[1] != "txt") return 0;

      let s = fs.readFileSync(dirname+'/'+name).toString();
      process(name,s);
    });

    decompressRooms();

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

function processTile(game, block)
{
  const name = block.split(/\s/)[1];

  let data = block.split('\n');
    data.shift();
    data = data.splice(0,8);

    if(data[0].length > 8)
    {
      console.log("Warning ! Game "+game+" has tiles resolution > 8x8. Consider removing it.")
    }


    let bin = [];

    for(let i in data)
    {
      bin[i] = [];
      for(let j in data[i])
      {
        bin[i][j] = parseInt(data[i][j]);
      }
    }

    save.tiles[game+"@@@"+name] = bin;

}

function processRoom(game, block)
{

  const name = block.split(/\s/)[1];

  let data = block.split('\n');
    data.shift();
    data = data.splice(0,16);

    let names = [];

    for(let i in data)
    {
      names[i] = [];
      let line = data[i].split(',');
      for(let j in line)
      {
        let tileName = line[j];
        names[i][j] = tileName;
      }
    }

    save.roomCompressed[game+"@@@"+name] = names;

    //Add the 0 empty tile for every room
    save.tiles[game+"@@@0"] = Array(8).fill(Array(8).fill(0));

}


function decompressRooms()
{
  for(let name in save.roomCompressed)
  {
    let game = name.split("@@@")[0];
    let room = save.roomCompressed[name];

    let roomBin = [];

    for(var gx = 0; gx < 16; gx++)
    {

      for(var gy = 0; gy < 16; gy++)
      {
        let tileName = room[gx][gy];
        let tile = save.tiles[game+"@@@"+tileName];
        for(var x = 0; x < 8; x++)
        {
          if(!roomBin[gx*8+x])roomBin[gx*8+x] = [];
          for(var y = 0; y < 8; y++)
          {
              if(!tile)console.log("Tile not found at "+game+"@@@"+tileName)

           roomBin[gx*8+x][gy*8+y] = tile[x][y];
          }
        }
      }
    }

    save.rooms.push(roomBin);

  }
}

function write()
{
	  //fs.writeFileSync(writename+'/'+i+'.json',JSON.stringify(save[i]));
	  //console.log("Wrote "+i+".json");

    fs.writeFileSync(writename+'/tiles.json',JSON.stringify(save.tiles));
    console.log("Wrote tiles.json");
    fs.writeFileSync(writename+'/rooms.json',JSON.stringify(save.rooms));
    console.log("Wrote rooms.json");
	  //console.log("Wrote "+i+".json");
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
console.log("Wrote avatars.json (from picture)")
}
