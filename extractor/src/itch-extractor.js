const request = require("request");
const jsdom = require("jsdom");
const {JSDOM} = jsdom;
const fs = require('fs');
const readline = require('readline');

listen();


function listen()
{
  console.log("Copy paste URLs and hit enter. You can do it multiple times, even when the previous one is not finished.");

  const stdin = process.openStdin();
  stdin.addListener("data", function(d) {
      // note:  d is an object, and when converted to a string it will
      // end with a linefeed.  so we (rather crudely) account for that
      // with toString() and then trim()

      try
      {

        let url = d.toString().trim();
        console.log("Extracting from "+url);
        extractFromItch(url);
      }
      catch(e)
      {
        console.log(e);
      }

    });


}



function extractFromItch(url)
{
  request(url, function (error, response, body) {

    console.log("");
    console.log("Itch connection - "+url)
    console.log('statusCode:', response && response.statusCode); // Print the response status code if a response was received
    const dom = new JSDOM(body);
    let url2;

    const iframe = dom.window.document.getElementById("game_drop");
    if(iframe)
    {
        url2 = "http:"+iframe.src;
    }
    else
    {
      // /!\ Pas forcément le numéro 0, s'il y en a plusieurs
      const iframeData = dom.window.document.getElementsByClassName('iframe_placeholder')[0].dataset.iframe;
      const iframeParsed = new JSDOM(iframeData);

      url2 = "http:"+iframeParsed.window.document.getElementById("game_drop").src;

    }

    extractData(url2);

  });
}


function extractData(url)
{
  request(url, function (error, response, body) {

    console.log("Direct connection - "+url)
    //console.log('statusCode:', response && response.statusCode); // Print the response status code if a response was received

    const dom = new JSDOM(body);
    const div =dom.window.document.getElementById("exportedGameData");

    if(div)
    {
       saveData(div.innerHTML);
    }
    else
    {
      console.log("ERROR : Could not find game data for "+url)
    }


  });
}


function saveData(data)
{
  let name = data.split("\n")[1].trim(); //extract name
  name = name.replace(/[^a-z0-9]/gi, '-').toLowerCase(); // safe_name
  if(name.length > 32)
  {
    name = name.substr(0,32);
  }

  fs.writeFile("./extracted/"+name+".txt", data, function(err) {
      if(err) {
          return console.log(err);
      }

      console.log("Saved "+name);
  });
}
