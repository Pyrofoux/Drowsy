const fs = require('fs');
let dirname = './extracted';
let writename = './formatted';

read();

let save = {};

save.avatars = [];
save.sprites = [];

function read()
{
  fs.readdir(dirname, (err, files) =>
  {
    files.forEach(name =>
    {
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

  if(name == 'A') // Avatar
  {
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

    save.avatars.push(bin);
  }

  //REFACTOR
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
	for(i in save)
	{
	  fs.writeFileSync(writename+'/'+i+'.json',JSON.stringify(save[i]));
	  console.log("Wrote "+i+".json");
	}
}
