// modified from
// https://stackoverflow.com/questions/12150491/toc-list-with-all-classes-generated-by-automodule-in-sphinx-docs

$(function (){
  var createList = function(selected) {
    var obj = {};
    var ul = $('<ul>');

    selected.each(function(i,e) {

      var groupName = $(e).find('a')[0].href;
      groupName = groupName.substr(groupName.lastIndexOf('/')+1);
      groupName = groupName.substr(0,groupName.lastIndexOf(".html"));


      var fullname = e.id;
      if (fullname.startsWith('tensorpack.'))
        fullname = fullname.substr(11);

      var n = $(e).children('.descname').clone();
      n[0].innerText = fullname;

      var l = $(e).children('.headerlink');
      var a = $('<a>');
      a.attr('href', l.attr('href')).attr('title', 'Link to this definition');
      a.append(n);

      var entry = $('<li>').append(a);

      if(groupName in obj) {
        obj[groupName].append(entry);
      } else {
        var ul = $('<ul style="margin-bottom: 12px;">');
        ul.append(entry);
        obj[groupName] = ul;
      }
    });

    return obj;
  }



  var customIndex = $('.custom-index');
  customIndex.empty();


  var selected = $('div.section>dl>dt');
  if (selected.length === 0)
    return;

  var obj = createList(selected);
  var block = $('<div style="min-width: 300px; margin-bottom: 2em;">');
  for(var key in obj) {
    var a = $('<h6 style="margin-bottom: 0;">');
    a.html(key + ':');
    block.append(a);
    block.append(obj[key]);
  }
  customIndex.append(block);
});
