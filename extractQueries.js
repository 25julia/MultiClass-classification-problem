/* this script is written in node js; 
It reads a json file 'train_intent_full.json" under folder Open_Source_Snips_Data/intentName/train_intent_full.json, in this case SearchScreeningEvent
Run this file for each intent to generate the concatenated queries
*/
var fs = require("fs");
console.log("\n *STARTING* \n");
// Get content from file
var contents = fs.readFileSync("train_SearchScreeningEvent_full.json");
var jsonContent = JSON.parse(contents);
var arrayOfData = jsonContent["SearchScreeningEvent"]
var array_query = [];
for (var i in arrayOfData) {
    var array = arrayOfData[i].data;
    var length = arrayOfData[i].data.length;
    var one_query = '';
    for (var k = 0; k < length; k++) {
        one_query += array[k].text;
    }
    array_query.push(one_query)
}

fs.writeFile('Search_Screening_Event_query.txt', array_query.join("\n"), (err) => {
    // throws an error, you could also catch it here
    if (err) throw err;
    // success case, the file was saved
    console.log('User query saved!');
});