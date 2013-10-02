document.addEventListener('click', textMining, true);

var bayes;

bayes = new classifier.Bayesian();
bayes.backend.catCounts = JSON.cats;
bayes.backend.wordCounts = JSON.words;

function textMining(eventData){
  var target = eventData.target;
  var innerText = target.innerText;

  var cat = bayes.classify(innerText);
  alert(cat + ": " + innerText);

}
