"""Published Snowball English stopwords, frozen September 16, 2026.

Source: https://snowballstem.org/algorithms/english/stop.txt
License: https://snowballstem.org/license.html
The full published list is tokenized using the search query's same word regex;
no benchmark questions, labels or query-specific words are added.

Copyright (c) 2001, Dr Martin Porter,
Copyright (c) 2002, Richard Boulton.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

import re

_PUBLISHED_WORDS = (
    "i me my myself we our ours ourselves you your yours yourself "
    "yourselves he him his himself she her hers herself it its itself "
    "they them their theirs themselves what which who whom this that these "
    "those am is are was were be been being have has had "
    "having do does did doing would should could ought i'm you're he's "
    "she's it's we're they're i've you've we've they've i'd you'd he'd she'd "
    "we'd they'd i'll you'll he'll she'll we'll they'll isn't aren't wasn't weren't "
    "hasn't haven't hadn't doesn't don't didn't won't wouldn't shan't shouldn't can't cannot "
    "couldn't mustn't let's that's who's what's here's there's when's where's why's how's "
    "a an the and but if or because as until while of "
    "at by for with about against between into through during before after "
    "above below to from up down in out on off over under "
    "again further then once here there when where why how all any "
    "both each few more most other some such no nor not only "
    "own same so than too very "
)
SNOWBALL_ENGLISH = frozenset(re.findall(r"\w+", _PUBLISHED_WORDS))
