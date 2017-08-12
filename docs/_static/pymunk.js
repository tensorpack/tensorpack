// modified from
// https://stackoverflow.com/questions/12150491/toc-list-with-all-classes-generated-by-automodule-in-sphinx-docs

$(function (){
  var createList = function(selected) {
    var ul = $('<ul>');

    selected.clone().each(function(i,e) {

      var n = $(e).children('.descname');
      var l = $(e).children('.headerlink');

      var a = $('<a>');
      a.attr('href',l.attr('href')).attr('title', 'Link to this definition');

      a.append(n);

      var entry = $('<li>').append(a);
      ul.append(entry);
    });
    return ul;
  }



  var customIndex = $('.custom-index');
  customIndex.empty();

  var mod_content = $('#module-contents');
  if (mod_content.length == 0)
    return;
  var mc_container = mod_content[0].parentNode;
  var selected = $(mc_container).find('>dl>dt');

  if (selected.length === 0)
    return;

  var l = createList(selected);

  var c = $('<div style="min-width: 300px;">');
  var ul = c.clone()
    .append(l);
  customIndex.append(ul);

});
