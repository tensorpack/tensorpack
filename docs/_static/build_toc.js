// modified from
// https://stackoverflow.com/questions/12150491/toc-list-with-all-classes-generated-by-automodule-in-sphinx-docs

$(function (){
  var createList = function(selected) {
    var ul = $('<ul>');

    selected.each(function(_, e) {
      var fullname = e.id;
      if (fullname.startsWith('tensorpack.'))
        fullname = fullname.substr(11);

      var n = $(e).children('.descname').clone();
      n[0].innerText = fullname;
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

  var selected = $('div.section>dl>dt');
  if (selected.length === 0)
    return;

  var l = createList(selected);

  var c = $('<div style="min-width: 300px;">');
  var ul = c.clone()
    .append(l);
  customIndex.append(ul);

});