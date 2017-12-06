//File: sanitize_desc_name.js

$(function (){
  var selected = $('div.section>dl>dt>code.descclassname');
  selected.each(function(_, e) {
    var text = e.innerText;
    if (text.startsWith('tensorpack.')) {
      text = text.substr(11);
      e.innerText = text;
    }
  });
});
