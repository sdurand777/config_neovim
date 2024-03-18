local ls = require("luasnip") --{{{
local s = ls.s --> snippet
local i = ls.i --> insert node
local t = ls.t --> text node

local d = ls.dynamic_node
local c = ls.choice_node
local f = ls.function_node
local sn = ls.snippet_node

local fmt = require("luasnip.extras.fmt").fmt
local rep = require("luasnip.extras").rep

local snippets, autosnippets = {}, {} --}}}

local group = vim.api.nvim_create_augroup("type de fichier latex ", { clear = true })
local file_pattern = "*.tex"

local function cs(trigger, nodes, opts) --{{{
    local snippet = s(trigger, nodes)
    local target_table = snippets

    local pattern = file_pattern
    local keymaps = {}

    if opts ~= nil then
        -- check for custom pattern
        if opts.pattern then
            pattern = opts.pattern
        end

        -- if opts is a string
        if type(opts) == "string" then
            if opts == "auto" then
                target_table = autosnippets
            else
                table.insert(keymaps, { "i", opts })
            end
        end

        -- if opts is a table
        if opts ~= nil and type(opts) == "table" then
            for _, keymap in ipairs(opts) do
                if type(keymap) == "string" then
                    table.insert(keymaps, { "i", keymap })
                else
                    table.insert(keymaps, keymap)
                end
            end
        end

        -- set autocmd for each keymap
        if opts ~= "auto" then
            for _, keymap in ipairs(keymaps) do
                vim.api.nvim_create_autocmd("BufEnter", {
                    pattern = pattern,
                    group = group,
                    callback = function()
                        vim.keymap.set(keymap[1], keymap[2], function()
                            ls.snip_expand(snippet)
                        end, { noremap = true, silent = true, buffer = true })
                    end,
                })
            end
        end
    end

    table.insert(target_table, snippet) -- insert snippet into appropriate table
end --}}}


-- Ecrire ses snippets lua on peut utiliser le snipet luasnippet 
cs("latextestsnippet", fmt( -- python test snippet
[[Latex test snippet;]], {}))


cs("ivmtemp", fmt( -- ivm pdf template
[=[
\documentclass[]{{paper}}
\usepackage[a4paper, total={{7in, 9in}}]{{geometry}}
\usepackage{{longfbox}}
\renewcommand\familydefault{{\sfdefault}}
% Useful packages
\usepackage{{amsmath,amsfonts}}
\usepackage[colorlinks=true, allcolors=blue]{{hyperref}}
% header
\usepackage{{fancyhdr}}
\pagestyle{{fancy}}
\fancyhead{{}}
\fancyhead[R]{{IVM Technologies - Projet NARVAL~- Page~\thepage }} 
\renewcommand{{\headrulewidth}}{{0.2pt}} % no line in header area
% footer
\fancyfoot{{}} % clear all footer fields
% page number in "outer" position of footer line
\fancyfoot[C]{{\scriptsize \indent All our business relations are regulated by our general conditions of sale, communicable on request. This document is the property of IVM TECHNOLO  GIES.
Anyone holding a copy acknowledges being bound by an obligation of secrecy and undertakes not to reproduce it, to communicate it to third parties in whole or in part,
or to use it for personal purposes, without written authorization from IVM TECHNOLOGIES.}}
\fboxset{{padding-left=6pt,padding-right=6pt,padding-top=12pt,padding-bottom=12pt}}%

\usepackage{{tikz}}

\usepackage[french]{{babel}}
\usepackage{{datetime}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}

\title{{\huge Compte Rendu \today : {} }}
\author{{Auteur : {} }}
\subtitle{{\huge {} }}

\usepackage{{fontsize}}
\changefontsize[17]{{17}}
  
\begin{{document}}
  
\maketitle
  
\begin{{tikzpicture}}[remember picture,overlay]
     \node[anchor=north east,inner sep=0pt] at (current page.north east)
              {{\includegraphics[scale=0.5]{{IVM}}}};
\end{{tikzpicture}}
  
\section{{ }}
\end{{document}}

]=], {
  i(1, "title"),
  i(2, "author"),
  i(3, "subtitle"),
  }))



cs("texsimple", fmt( -- simple document latex
[[
\documentclass[12pt]{{article}}
\begin{{document}}
Test
\end{{document}}
{}
]], {
    i(1,""),
  }))



cs("envtest", fmt( -- New latex env
[[
\begin{{{doc}}}
{}
\end{{{doc}}}
]], {doc=i(1,""),i(2,"texter")},
{repeat_duplicates=true}))
--  {
--    repeat_duplicates = true
--  }))












cs("diagramme", fmt( -- diagramme
[[
\begin{{figure}}[!htb]
\centering
\begin{{tikzpicture}}[
>=latex',
auto
]

\node [intg] (kp)  {{Sun/Radiation}};
\node [int]  (ki1) [node distance=1.5cm and -1cm,below left=of kp] {{Methane/Carbon Dioxide/Burning of Fossil Fuels}};
\node [int]  (ki2) [node distance=1.5cm and -1cm,below right=of kp] {{Deforestation/Melting Ice/Chemical Induced Fertilizer  }};
\node [intg] (ki3) [node distance=5cm,below of=kp] {{More Storms and Failing Crops}};
\node [intg] (ki4) [node distance=2cm,below of=ki3] {{Extinction}};

\draw[->] (kp) -- ($(kp.south)+(0,-0.75)$) -| (ki1) node[above,pos=0.25] {{}} ;
\draw[->] (kp) -- ($(kp.south)+(0,-0.75)$) -| (ki2) node[above,pos=0.25] {{}};
\draw[->] (ki1) |- (ki3);
\draw[->] (ki2) |- (ki3);
\draw[->] (ki3) -- (ki4);
\end{{tikzpicture}}
\caption{{}}
\label{{}}
\end{{figure}}
]], {}))





cs("diagrammesetup", fmt( -- setup pour construire des diagrammes avec latex 
[=[
\documentclass[]{{paper}}
\usepackage[T1]{{fontenc}}
\usepackage{{tikz}}
\usetikzlibrary{{arrows,calc,positioning}}
\tikzstyle{{intt}}=[draw,text centered,minimum size=6em,text width=5.25cm,text height=0.34cm]
\tikzstyle{{intl}}=[draw,text centered,minimum size=2em,text width=2.75cm,text height=0.34cm]
\tikzstyle{{int}}=[draw,minimum size=2.5em,text centered,text width=3.5cm]
\tikzstyle{{intg}}=[draw,minimum size=3em,text centered,text width=6.cm]
\tikzstyle{{sum}}=[draw,shape=circle,inner sep=2pt,text centered,node distance=3.5cm]
\tikzstyle{{summ}}=[drawshape=circle,inner sep=4pt,text centered,node distance=3.cm]
]=], {}))




cs("ppttemplate", fmt( -- template ppt 
[[
\documentclass[10pt]{{beamer}}
\usetheme{{Boadilla}}

% paquet images deplacement
\usepackage{{eso-pic}}

% paquets pour le français
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}

\title{{ {} }}
\author{{ {} }}
\institute{{ {} }}

%\logo{{
%  \includegraphics[height=0.75cm]{{IVM.png}}
%}}

%\logo{{\makebox[1\paperwidth]{{\includegraphics[width=5cm,keepaspectratio]{{IVM.png}}}}}}

\newcommand\AtPagemyUpperLeft[1]{{\AtPageLowerLeft{{%
\put(\LenToUnit{{0.75\paperwidth}},\LenToUnit{{0.9\paperheight}}){{#1}}}}}}
\AddToShipoutPictureFG{{
\AtPagemyUpperLeft{{{{\includegraphics[width=3cm,keepaspectratio]{{IVM.png}}}}}}
}}%

\begin{{document}}

% frame titre
\frame{{\titlepage}}


\end{{document}}
]], {
    i(1,"title"),
    i(2,"author"),
    i(3,"institute")
}))


cs("codesetup", fmt( -- setup package pour ecrire du code dans latex 
[[
\usepackage{{color}}
\usepackage{{listings}}
\lstset{{language=C++,
	  basicstyle=\ttfamily,
	  keywordstyle=\color{{blue}}\ttfamily,
	  stringstyle=\color{{red}}\ttfamily,
	  commentstyle=\color{{green}}\ttfamily,
	  morecomment=[l][\color{{magenta}}]{{\#}}
}}
]], {}))


cs("codabloc", fmt( -- bloc pour écrire du code 
[[
\begin{{lstlisting}}
{}
\end{{lstlisting}}
{}
]], {i(1,"code remplir"),i(2,"fin du bloc")
}))




cs("doctexsimple", fmt( -- simple document latex
[[
\documentclass[12pt]{{article}}
\begin{{document}}
\end{{document}}
{}
]], {
    i(1,""),
  }))



cs("french", fmt( -- snippet pour avoir le français sous latex 
[[
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage[french]{{babel}}
]], {}))


cs("frenchdate", fmt( -- package pour avoir la date en francais sous latex 
[[
\usepackage[french]{{babel}}
\usepackage{{datetime}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
]], {}))

cs("titreoptions", fmt( -- snippet pour le titre 
[[
\title{{Compte Rendu \today : {} }}
\author{{Author : {} }}
\subtitle{{ {} }}
]], {i(1,""),i(2,""),i(3,"")}))

cs("margin", fmt( -- snippet pour ajuster la marge du document 
[[
% enter width and height to write your document
\usepackage[a4paper, total={{7in, 9in}}]{{geometry}}
]], {}))

cs("fontsize", fmt( -- snippet pour ajuster le fontsize 
[[
% enter space line and fontsize 17 is large
\usepackage{{fontsize}}
  \changefontsize[ {size} ]{{ {size} }}
]], {size=i(1,"")},{repeat_duplicates=true}))


cs("vector_vertical", fmt( -- snippet pour afficher un vecteur vertical 
[[
\begin{{bmatrix}} V_x \\ V_y \\ V_z \end{{bmatrix}}
]], {
  }))

cs("vector_horizontal", fmt( -- snippet pour afficher un vecteur vertical 
[[
\begin{{bmatrix}} V_x & V_y & V_z \end{{bmatrix}}
]], {
  }))


cs("matrix_33", fmt( -- snippet pour une matricce simple en latex 
[[
\begin{{bmatrix}} a & b & c \\ d & e & f \\ g & h & i \end{{bmatrix}}
]], {
  }))

cs("matrix_22", fmt( -- snippet pour une matricce simple en latex 
[[
\begin{{bmatrix}} a & b \\ c & d \end{{bmatrix}}
]], {
  }))

cs("matrix_44", fmt( -- snippet pour une matricce simple en latex 
[[
\begin{{bmatrix}} a & b & c & d \\ d & e & f & g \\ g & h & i & j \end{{bmatrix}}
]], {
  }))

cs("wedge", fmt( -- snippet pour affihcer une puissance chapeau
[[
^{{\wedge}}
]], {}))

cs("prime", fmt( -- snippet pour affihcer une puissance chapeau
[[
^{{\prime}}
]], {}))

cs("star", fmt( -- snippet pour affihcer une puissance chapeau
[[
^{{\star}}
]], {}))


cs("minlim", fmt( -- snippet pour faire un min avec limits 
[[
\min\limits_{}
]], {
    i(1,"limit")
  }))

cs("normtex", fmt( -- snippet pour afficher norm 
[[
\|{}\|
]], {
    i(1,"")
  }))

cs("powertex", fmt( -- snippet pour afficher norm 
[[
^{{{}}}
]], {
    i(1,"")
  }))

cs("dottex", fmt( -- snippet pour afficher dot
[[
\dot{{{}}}
]], {
    i(1,"")
  }))

cs("undertex", fmt( -- snippet pour afficher norm 
[[
_{{{}}}
]], {
    i(1,"")
  }))




cs("tikzfig", fmt( -- snippet pour dessiner une figure 
[[
\begin{{figure}}[ht!]
\centering
\begin{{tikzpicture}}[
>=latex',
auto
]
{}
\end{{tikzpicture}}
\end{{figure}} 
]], {
    i(1,"start drawing")
  }))


cs("tikzdecoration", fmt( -- snippet pour embraces tikz 
[[
\draw [decorate,decoration={{brace,amplitude=5pt,raise=2ex}}] (-1.5,1.5) -- (-1.5,0) node[midway, xshift=0.5cm] {{u}}; 
]], {
  }))


cs("tikzequation", fmt( -- snippet pour node tikz equation 
[[
\draw (0,0) node[] {{${}$}};
]], {
    i(1,"type your equation")
  }))



cs("nodetex", fmt( -- simple node tex use the node intg define at the beginning 
[[
\node [intg] (kp2) [right=8cm of kp1, align=left]
{{
    {}   
}};
]], {
  i(1,"node text")
  }))


cs("drawnodetex", fmt( -- simple draw node tex 
[[
\draw (0,0) node{{{}}};
]], {
  i(1,"write node text")
  }))


cs("rectanglenodetex", fmt( -- simple rectangle node tex
[[
\draw (0,0) + (-1,-1) rectangle +(1,1);
]], {
  }))


cs("gridnodetex", fmt( -- simple grid node tex
[[
\foreach \y in {{0,...,-2}}
    \foreach \x in {{0,...,2}}
    {{
        \draw (\x,\y) + (-.5,-.5) rectangle ++(.5,.5);
        \draw (\x,\y) node{{\the\numexpr\y*(-2)+\x}};
    }}
]], {
  }))


cs("conditionalnodetex", fmt( -- condition in tikz node 
[[
% Utilisez l'opérateur ternaire pour décider de la valeur à afficher
\pgfmathtruncatemacro{{\value}}{{
(\x==0 || \y==0 || \x==4 || \y==-4) ? 0 : \y*(-3)+\x -4
}}
\draw (\x,\y) node{{\value}};
]], {
  }))


cs("real_tex", fmt( -- real symbole latex 
[[
\mathbb{{R}} 
]], {
  }))

cs("gaussian_tex", fmt( -- gaussian symbole latex 
[[
\mathcal{{N}} 
]], {
  }))


cs("rectangle_node_tex", fmt( -- rectangle latex
[[
\node[rectangle,draw] (r) at (0,0) {{}};
]], {
  }))

cs("circle_node_tex", fmt( -- rectangle latex
[[
\node[circle,draw] (c) at (0,0) {{}};
]], {
  }))


cs("arrow_node_tex", fmt( -- simple arrow node latex 
[[
\draw[->] ($(kp1)$) -- ($(kp2)$);
]], {
  }))


cs("times_tex", fmt( -- times symbol latex 
[[
\times
]], {
  }))


cs("arrow_corner_node", fmt( -- corner arrow latex 
[[
\draw[->] ($(r2.north)$) |- ($(r1.west)$);
]], {
  }))


cs("node_with_id", fmt( -- simple node latex avec id
[[
\node (id) at (0,0) {{texte}};
]], {
  }))


cs("bend_arrow_node_tex", fmt( -- bended arrow latex 
[[
\draw[->] ($(r7.north)$) to[bend left] ($(r1.west)$);
]], {
  }))

-- Tutorial Snippets go here --

-- End Refactoring --

return snippets, autosnippets

