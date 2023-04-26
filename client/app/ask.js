// common.js

function getUrlParams() {
    // Ref: https://stackoverflow.com/questions/4656843/get-querystring-from-url-using-jquery/4656873#4656873
    var vars = [], hash;
    var hashes = window.location.href.slice(window.location.href.indexOf('?') + 1).split('&');
    for(var i = 0; i < hashes.length; i++) {
        hash = hashes[i].split('=');
        vars.push(hash[0]);
        vars[hash[0]] = hash[1];
    }
    return vars;
}

function setUrlQueryValue(paramKey, paramVal) {
    try {
        var href = new URL(location.href);
        // href.searchParams.set(paramKey, encodeURI(paramVal));  // NOTE: double encoding!!
        href.searchParams.set(paramKey, paramVal);  // Encoded, like with encodeURI(), except + instead of %20 which is annoying
        window.history.pushState("", "", href);
        return encodeURIComponent(paramVal);  //.replace(/%20/g, '+');   // TODO: review, replace not needed?
    } catch(err) { }
}

function deleteUrlQueryKey(paramKey) {
    try {
        var href = new URL(location.href);
        href.searchParams.delete(paramKey);
        window.history.pushState("", "", href);  // TODO: review
    } catch(err) { }
}

// Options: s (style), l (logo), d (debug), b (business)
function getUrlQuery(k) {
    try {
        var params = getUrlParams();
        if (k in params) {
            // NOTE: plus_fixed replace to solve href.searchParams.set encoding as described above.
            //       https://stackoverflow.com/questions/4535288/why-doesnt-decodeuriab-a-b/4535319#4535319
            var plus_fixed = (params[k]+'').replace(/\+/g, '%20')
            return decodeURIComponent(plus_fixed); } else { return "";
        }
    } catch(err) {
        return "";
    }
}


function toTitleCase(str) {
  return str.replace(
    /\w\S*/g,
    function(txt) {
      return txt.charAt(0).toUpperCase() + txt.substr(1).toLowerCase();
    }
  );
}

// https://stackoverflow.com/questions/400212/how-do-i-copy-to-the-clipboard-in-javascript/33928558#33928558
function copyToClipboard(text) {
    if (window.clipboardData && window.clipboardData.setData) {
        // Internet Explorer-specific code path to prevent textarea being shown while dialog is visible.
        return window.clipboardData.setData("Text", text);

    }
    else if (document.queryCommandSupported && document.queryCommandSupported("copy")) {
        var textarea = document.createElement("textarea");
        textarea.textContent = text;
        textarea.style.position = "fixed";  // Prevent scrolling to bottom of page in Microsoft Edge.
        document.body.appendChild(textarea);
        textarea.select();
        try {
            return document.execCommand("copy");  // Security exception may be thrown by some browsers.
        }
        catch (ex) {
            console.warn("Copy to clipboard failed.", ex);
            return prompt("Copy to clipboard: Ctrl+C, Enter", text);
        }
        finally {
            document.body.removeChild(textarea);
        }
    }
}

function scrollToBottom() {
    // TODO: investigate smooth: https://stackoverflow.com/questions/42261524/how-to-window-scrollto-with-a-smooth-effect
    window.scrollTo(0, $(document).height());
}


$(document).ready(function () {
    $("#menu-icon").click(function(){
        // https://www.w3schools.com/howto/howto_js_sidenav.asp
        $('#sidenav').width("200px");
    });

    $("#menu-close-btn").click(function(){
        $('#sidenav').width("0px");
    });

    $("#container").click(function(){
        $('#sidenav').width("0px");
    });
});


// ask.js

const app = "awepy-ask-202304";
const baseAppUrl = "https://www.awesomepython.org";
const authkey = "58906b3d-4a8a-4712-89c8-56446590ef73";
const predictionUrl = "https://chat-api.infocruncher.com/predict/";
const randomInputs = [
        "Format a date string in Python with format YYYMMDD",
        "Optimize my Python code for performance",
        "Read and write files in Python",
        "Work with dictionaries and lists in Python",
        "Handle exceptions (try, except) in Python",
        "Install and use third-party libraries in Python using pip",
        "Work with strings and manipulate text in Python",
        "Handle and raise custom exceptions in Python",
        "Work with classes, inheritance, polymorphism, and objects in Python",
        "Handle dates, times, and timezones using the Python datetime module",
        "Perform file I/O operations with CSV, JSON, XML, and binary files",
        "Work with databases using SQLAlchemy, SQLite3, and executing queries",
        "Import and organize code using modules, packages, and namespaces",
        "Use regular expressions (regex) for text pattern matching and manipulation",
        "Implement concurrent programming with threading, multiprocessing, and asynchronous programming?",
        "Reverse a string in Python",
        "Find the length of a list in Python",
        "Read a CSV file in Python",
        "Generate random numbers in Python",
        "Check if a file exists in Python",
        "Calculate the factorial of a number in Python",
        "Convert a string to lowercase in Python",
        "Concatenate two lists in Python",
        "Remove duplicates from a list in Python",
        "Install and use a third-party library in Python",
        "Find the maximum value in a dictionary in Python",
        "Sort a list of integers in ascending order in Python",
        "Calculate the square root of a number in Python",
        "Read and write JSON files in Python",
        "Remove whitespace from the beginning and end of a string in Python",
        "Count the occurrences of a specific element in a list in Python",
        "Format a date in a specific format using datetime module in Python",
        "Make a POST request to a RESTful API in Python",
        "Encrypt and decrypt data using the cryptography library in Python",
        "Implement a stack data structure in Python",
        "Capture and handle exceptions in Python",
        "Check if a string contains a substring in Python",
        "Convert a list of strings to a single string in Python",
        "Use regular expressions to search and manipulate strings in Python",
        "Write unit tests for Python code using the built-in unittest module",
        "Read and write binary files in Python",
        "Implement a binary search algorithm in Python",
        "Calculate the mean, median, and mode of a list of numbers in Python",
        "Parse and extract data from XML files in Python",
        "Implement multithreading in Python to improve performance",
        "Implement a queue data structure in Python",
        "Iterate over a dictionary in Python",
        "Use the logging module for logging messages in Python",
        "Implement a bubble sort algorithm in Python",
        "Read and write Excel files in Python using a library like pandas",
        "Use the os module to interact with the operating system in Python",
        "Implement a binary tree data structure in Python",
        "Use the argparse module to parse command-line arguments in Python",
        "Generate a random password with a given length in Python",
        "Implement memoization to optimize recursive functions in Python",
        "Implement a decorator in Python",
        "Use context managers with the 'with' statement in Python",
        "Serialize and deserialize Python objects using the pickle module",
        "Implement a queue data structure with priority in Python",
        "Use the itertools module for advanced iteration in Python",
        "Implement a depth-first search algorithm for graph traversal in Python",
        "Use the timeit module to measure the performance of Python code",
        "Implement a LRU (Least Recently Used) cache in Python",
        "Use the subprocess module to run external commands in Python",
        "Implement a custom exception class in Python for error handling",
        "Implement a binary search tree data structure in Python",
        "Use the contextlib module for creating context managers in Python",
        "Implement a breadth-first search algorithm for graph traversal in Python",
        "Use the multiprocessing module for parallel processing in Python",
        "Implement memoization using functools.lru_cache in Python",
        "Use the shutil module for file and directory operations in Python",
        "Implement a merge sort algorithm in Python",
        "Use the sqlite3 module for SQLite database operations in Python",
        "Implement a circular linked list data structure in Python",
        "Use the difflib module for text comparison and difference detection in Python",
        "Implement a priority queue data structure in Python",
        "Use the concurrent.futures module for concurrent programming in Python",
        "Implement a decorator with arguments in Python",
        "Use the enum module for creating enumerations in Python",
        "Implement a radix sort algorithm in Python",
        "Use the logging module to log messages to a file in Python",
        "Implement a Dijkstra's algorithm for finding shortest path in Python",
        "Use the itertools module for permutations and combinations in Python",
        "Implement a heap data structure in Python",
        "Use the threading module for thread-based concurrency in Python",
        "Implement a Trie data structure in Python for efficient string operations",
        "Use the math module for mathematical calculations in Python",
        "Implement a Fisher-Yates shuffle algorithm in Python for shuffling lists",
        "Use the built-in collections module in Python for advanced data manipulation",
        "Implement a bit manipulation operation, such as bitwise AND or XOR, in Python",
        "Use the re module for regular expression operations in Python",
        "Implement a dynamic programming algorithm in Python for solving optimization problems",
        "Use the concurrent.futures module for parallelism with ThreadPoolExecutor in Python",
        "Implement a Bloom filter data structure in Python for probabilistic set membership testing",
        "Use the built-in statistics module in Python for statistical calculations, such as mean, median, and variance",
        "Implement a merge operation in merge sort in Python",
        "Use the collections module to implement a deque (double-ended queue) in Python",
        "Implement a sliding window algorithm for processing arrays or strings in Python",
        "Use the csv module for reading and writing CSV files in Python",
        "Implement a selection sort algorithm in Python for sorting lists",
        "Use the time module for measuring time intervals and delays in Python",
        "Implement a Floyd-Warshall algorithm for finding all-pairs shortest path in Python",
        "Use the built-in functools module in Python for higher-order functions and decorators",
        "Implement a union-find data structure in Python for disjoint set operations",
        "Use the Python standard library for working with dates, times, and timezones",
        "Import and use a Python package",
        "Install a Python package using pip",
        "Upgrade a Python package to the latest version",
        "Uninstall a Python package using pip",
        "Use a Python package for data visualization",
        "Use a Python package for working with databases",
        "Use a Python package for web scraping",
        "Use a Python package for machine learning",
        "Use a Python package for working with APIs",
        "Use a Python package for handling dates and times",
        "Use a Python package for handling regular expressions",
        "Use a Python package for working with scientific computing",
        "Use a Python package for working with images and multimedia",
        "Use a Python package for working with natural language processing",
        "Use a Python package for working with network programming",
        "Use a Python package for working with data serialization and deserialization",
        "Use a Python package for working with encryption and decryption",
        "Use a Python package for working with web development and APIs",
        "Use a Python package for working with data analysis and visualization",
        "Use a Python package for working with machine learning model evaluation and performance metrics"
    ]

var rndIndex = Math.floor(Math.random() * randomInputs.length);
var predict_count = 0;

function populateQueryShareLink(businessIdea) {
    var shareLinkValue = baseAppUrl;
    if (businessIdea != "") {
        shareLinkValue += "?b=" + encodeURIComponent(businessIdea);
    }
//    if (businessIdea != "" && style != "") {
//        shareLinkValue += "&s=" + encodeURIComponent(style);  // TODO: need to select radio button of style
//    }
    $("#copy-idea-button").data("link", shareLinkValue);
}

var bar = new ProgressBar.Line(
    '#progress',
    {
        easing: 'easeInOut',
        duration: 13000,
        color: '#3776ab',
    });

function showProgressBar() {
    // TODO: dynamically set duration based on style
    $('#progress').show();
    bar.set(0.01);
    bar.animate(1, {}, function() {
        var existingText = $("#response").html();
        $("#response").html(existingText + "<p><i>Doing some fancy formatting, soooooooooooo close now!...</i></p>");
    });
}

function hideProgressBar() {
    bar.set(1);
    $('#progress').hide("slow");
    bar.set(0);
}

$(document).ready(function () {
    var businessIdea = getUrlQuery("b");
    var style = getUrlQuery("s");
    var load = getUrlQuery("load");
    console.log("User querystring businessIdea: " + businessIdea);
    console.log("User querystring style: " + style);
    console.log("User querystring load: " + load);

    $("#input-text").focus();
    populateQueryShareLink(businessIdea);

    if (load != "") {
        // Load a previously generated report to display
        $("#input-text").val("load:" + load);
        // setTimeout(function(){ $('#input-button').click()}, 100);
        processReport();

    } else if (businessIdea != "") {
        // Nothing to load, user supplied a business idea in querystring
        $("#input-text").val(businessIdea);
        populateQueryShareLink(businessIdea);

//    } else {
//        // Autopopulate input with a random input
//        var randomElement = randomInputs[Math.floor(Math.random() * randomInputs.length)];
//        $("#input-text").val(randomElement);
//        populateQueryShareLink(randomElement);
    }

    $("#clear-button").click(function(){
        deleteUrlQueryKey("b");
        $("#input-text").val("")
        $("#input-text").focus();
        populateQueryShareLink("");
    });

    $("#random-button").click(function(){
        rndIndex += 1;
        if (rndIndex >= randomInputs.length) {
            rndIndex = 0;
        }
        var randomElement = randomInputs[rndIndex];
        $("#input-text").val(randomElement);
        populateQueryShareLink(randomElement);
    });

    $("#view-option-info").click(function(){
        $("#share-info").hide();
        $("#option-info").toggle();
        scrollToBottom();
        return false;
    });

    $("#view-option-info-intext").click(function(){
        $("#share-info").hide();
        $("#option-info").toggle();
        scrollToBottom();
        return false;
    });

    $("#view-share-info").click(function(){
        $("#option-info").hide();
        $("#share-info").toggle();
        scrollToBottom();
        return false;
    });

    $("#input-text").keypress(function (e) {
      if (e.which == 13) {
        $("#input-button").click();
        return false;
      }
    });

    $("#copy-idea-button").click(function() {
        var link = $(this).data("link");
        console.log("Copied idea link: " + link);
        copyToClipboard(link);

        var button = $(this);
        button.addClass("button-clicked");
        setTimeout(function() {
          button.removeClass("button-clicked");
        }, 200);
    });

    $("#copy-report-button").click(function() {
        var link = $(this).data("link");
        console.log("Copied report link: " + link);
        copyToClipboard(link);

        var button = $(this);
        button.addClass("button-clicked");
        setTimeout(function() {
          button.removeClass("button-clicked");
        }, 200);
    });

    $("#input-button-bottom").click(function(){
        document.documentElement.scrollTop = 0;
        return false;
    });

    $(".input-button-class").click(function() {  // NOTE: match class as there are 2 buttons
        processReport();
    });

    var start = new Date().getTime();
    $.ajax({
       type: "POST",
       url: predictionUrl,
       data: '{"app": "' + app + '", "authkey": "' + authkey + '", "query": "PrimeLambda", "mode": "PrimeLambda"}',
       tryCount : 0,
       retryLimit : 1,
       success: function(data)
       {
            predict_count++;
            var end = new Date().getTime();
            var time = end - start;
            var debugInfo = "\nPRIME SUCCESS:\nClient timing (ms): " + time + "\n"
                             + "Retries: "+this.tryCount+"\n"
                             + "Success Reponse:\n"
                             + JSON.stringify(data, null, "  ");
            console.log(debugInfo);
       },
       error: function(data)
       {
           this.tryCount++;
           if (this.tryCount <= this.retryLimit) {
               $("#response").html("Retry " + this.tryCount + "...");
               $.ajax(this);
               return;
           }
           var end = new Date().getTime();
           var time = end - start;

           var debugInfo = "\nPRIME ERROR:\nClient timing (ms): " + time + "\n"
                            + "Retries: "+this.tryCount+"\n"
                            + "Error Reponse:\n"
                            + JSON.stringify(data, null, "  ");
           console.log(debugInfo);
       },
     });
});


function processReport() {
    var raw_input = $("#input-text").val();
    var input = DOMPurify.sanitize(raw_input.replace(/(\r\n|\n|\r)/gm, " "));
    if (input.length == 0) {
        $("#response").html("You haven't entered any career dreams in the textbox above!<br /><br />"
            + "Enter the unicorn career do you want build (or try experimenting by clicking the 'Random Career' button) and give it another whirl");
        return;
    }

    var max_query_length = 250; // TODO: sync with app.py
    if (input.length > max_query_length) {
        $("#response").html("Can you squish your wild dream into " + max_query_length + " characters or less, and try again?<br />");
        return;
    }

    var mode = "section_dict";
    var style = getUrlQuery("s");  // NOTE: see `style_dict` in code for list.
    if (style == "") {
        style = $('input:radio[name="response_style"]:checked').val();
    }
    var logo = getUrlQuery("l");
    if (logo == "") {
        logo = $('input:radio[name="include_logo"]:checked').val();
    }
    var debug = getUrlQuery("d");

    var predictionUrlPayloadObject = {
        "app": app,
        "authkey": authkey,
        "query": input,
        "mode": mode,
        "style": style,
        "logo": logo,
        "debug": debug
    }
    var predictionUrlPayload = JSON.stringify(predictionUrlPayloadObject);
//        console.log("predictionUrlPayload: " + predictionUrlPayload);

    // Update URL with input value for sharing
    var paramVal = setUrlQueryValue("b", input);

    // Populate share link
    populateQueryShareLink(input);

    // Handle UI for input processing
    showProgressBar();  // New progress bar
    var originalInputButtonHtml = $("#input-button").html();
    var spinningInputButtonHtml = '<i class="fa fa-spinner fa-spin"></i> ' + originalInputButtonHtml;
    $("#input-button").html(spinningInputButtonHtml);
    $("#input-button").prop('disabled', true);
    $("#input-button-bottom").html(spinningInputButtonHtml);
    $("#input-button-bottom").prop('disabled', true);

    var style_text = style.length > 0 && style != "general" ? " in the style of " + toTitleCase(style) : "";  // Sync with business_report.py
    var running_text = "";
    if (input.startsWith("load:")) {
        // Loading a saved report
        running_text += "<p><em>Loading the answer to your question</em><p>";
    } else {
        // Building a saved report
        running_text += "<p><em>Asking an experienced rubber duck now" + style_text + "</em><p>";
        running_text += "<h2>" + input + "</h2>";
    }

    running_text += "<p><i>Working on it now...almost done...</i></p>";
    if (predict_count == 0) {
       $("#response").html(running_text);  // NOTE: possible Lambda fn isn't primed yet.
    } else {
       $("#response").html(running_text);  // NOTE: Lambda should be primed.
    }

    var start = new Date().getTime();
    $.ajax({
       type: "POST",
       url: predictionUrl,
       data: predictionUrlPayload,
       tryCount : 0,
       retryLimit : 1,
       success: function(data)
       {
           // Reset UI after processing (success)
           hideProgressBar();  // New progress bar
           $("#input-button").html(originalInputButtonHtml);
           $("#input-button").prop('disabled', false);
           $("#input-button-bottom").html(originalInputButtonHtml);
           $("#input-button-bottom").prop('disabled', false);

           predict_count++;
           var end = new Date().getTime();
           var time = end - start;
           var chat_response = data["result"]["chat_response"];
           var request_id = data["request_id"];
           var request_id_loaded = data["request_id_loaded"];
           var query = data["query"];

           // console.log("chat_response: " + chat_response);
           console.log("request_id: " + request_id);
           console.log("request_id_loaded: " + request_id_loaded);
           console.log("query: " + query);

           var permalinkHref = "/ask.html?load=" + request_id + "&desc=" + encodeURIComponent(query);
           var permalink = '<a href="' + permalinkHref + '">Permalink</a>';
           var html_response = "";
           style_text = (style == "general") ? "" : " in the style of " + toTitleCase(style);
           if (request_id_loaded != null && query != null) {
                // Request loaded an existing business plan, update things for this case
                $("#input-text").val(query);
                $("#share-info").show();
                deleteUrlQueryKey("b");
                html_response += "<i>Loaded your answer" + style_text + " (" + permalink + "):</i><br /><br />";

           } else {
                html_response += "<i>Here is your answer" + style_text + " (" + permalink + "):</i><br /><br />";
           }

           var markdown_as_sanitized_html = DOMPurify.sanitize(marked.parse(chat_response));  // Convert markdown chat_response into sanitized html
           html_response += markdown_as_sanitized_html;
           html_response += permalink + '<br /><br />';
           html_response += '<a href="#" onclick="document.documentElement.scrollTop = 0; return false;">Back to top</a>';

           $("#response").html(html_response);
           $("pre code").each(function(i, block) {
                // https://highlightjs.org/usage/
                // https://highlightjs.org/static/demo/
                hljs.highlightElement(block);
                // console.log("hljs applied to code: " + block);
           });

           var copyReportPermalink = baseAppUrl + permalinkHref;
           $("#copy-report-button").data("link", copyReportPermalink);
           $("#copy-report-button").show();
           $("#copy-idea-button").hide();
           $("#share-info").show();

//               var debugInfo = "\nDEBUG INFO:\nClient timing (ms): " + time + "\n"
//                                                    + "Retries: "+this.tryCount+"\n"
//                                                    + "Success Reponse:\n"
//                                                    + JSON.stringify(data, null, "  ");
//               console.log(debugInfo);
       },
       error: function(data)
       {
           this.tryCount++;
           if (this.tryCount <= this.retryLimit) {
               $("#response").html("Hmmm, server is a bit busy, let me try that again for you...retry " + this.tryCount + " of " + this.retryLimit + "...");
               $.ajax(this);
               return;
           }
           var end = new Date().getTime();
           var time = end - start;

           // Reset UI after processing (error)
           hideProgressBar();  // New progress bar
           $("#input-button").html(originalInputButtonHtml);
           $("#input-button").prop('disabled', false);
           $("#input-button-bottom").html(originalInputButtonHtml);
           $("#input-button-bottom").prop('disabled', false);

           $("#response").html("Server load is high at the moment<br /><br />Please try again...")

           var debugInfo = "\nDEBUG INFO:\nClient timing (ms): " + time + "\n"
                                        + "Retries: "+this.tryCount+"\n"
                                        + "Error Reponse:\n"
                                        + JSON.stringify(data, null, "  ");
           console.log(debugInfo);
       },
     });
}
