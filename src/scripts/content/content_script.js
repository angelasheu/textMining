document.addEventListener('click', textMining, true);

var bayes;

bayes = new classifier.Bayesian();

bayes.backend.catCounts = json.cats;
bayes.backend.wordCounts = json.words;

//alchemyTest();

function textMining(eventData){
  var target = eventData.target;
  var innerText = target.innerText;

  var cat = bayes.classify(innerText);
  //alert(cat + ": " + innerText);

}

function alchemyTest() {
	var sample_url = 'http://alvaradoschool.net/about/general-education-program/';
	keywords('url', sample_url, {}, function(error, response) {
		var keywords = response.keywords;
		for (var i = 0; i < keywords.length; i++) {
			console.log(keywords[i]);
		}
	})
}