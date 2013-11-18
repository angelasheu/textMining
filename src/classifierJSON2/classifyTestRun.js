// Currently only testing advisory


var fs = require('fs');
var classifier = require('classifier'), bayes;
var jsonString, jsonObj;

bayes = new classifier.Bayesian();
jsonString = fs.readFileSync('parent', 'utf8');
jsonObj = JSON.parse(jsonString);

bayes.fromJSON(jsonObj);

var cat = bayes.classify("PTA. How can we help? Parent teacher conference. Involvement with the school. Parent newsletter.");
console.log(cat);



