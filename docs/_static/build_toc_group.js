// modified from
// https://stackoverflow.com/questions/12150491/toc-list-with-all-classes-generated-by-automodule-in-sphinx-docs

$(function (){
  var createList = function(selected) {
    var obj = {};
    var ul = $('<ul>');

    selected.each(function(i,e) {
      
      var className = e.getElementsByTagName('a')[0].href;
      className = className.substr(className.lastIndexOf('/')+1);
      className = className.substr(0,className.lastIndexOf(".html"));



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

      if(className in obj) {
        obj[className] = obj[className].append(entry);
      } else {
        var ul = $('<ul>');
        ul.append(entry);
        obj[className] = ul;
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
  for(var key in obj) {
    var c = $('<div style="min-width: 300px;">');
    var a = $('<h4>');
    a.html(key);
    var u = c.clone().append(a);
    var ul = c.clone().append(obj[key]);
    customIndex.append(u);
    customIndex.append(ul);
  }

});
