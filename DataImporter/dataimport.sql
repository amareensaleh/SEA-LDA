create database BSE_bitcoin DEFAULT CHARACTER SET utf8 DEFAULT COLLATE utf8_general_ci;

use BSE_bitcoin;

CREATE TABLE `badges` (
  `Id` int(11) NOT NULL,
  `UserId` int(11) DEFAULT NULL,
  `Name` varchar(50) DEFAULT NULL,
  `Date` datetime DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;


CREATE TABLE `comments` (
  `Id` int(11) NOT NULL,
  `PostId` int(11) NOT NULL,
  `Score` int(11) NOT NULL DEFAULT '0',
  `Text` text,
  `CreationDate` datetime DEFAULT NULL,
  `UserId` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

CREATE TABLE `posts` (
  `Id` int(11) NOT NULL,
  `PostTypeId` smallint(6) DEFAULT NULL,
  `AcceptedAnswerId` int(11) DEFAULT NULL,
  `ParentId` int(11) DEFAULT NULL,
  `Score` int(11) DEFAULT NULL,
  `ViewCount` int(11) DEFAULT NULL,
  `Body` text,
  `OwnerUserId` int(11) NOT NULL,
  `OwnerDisplayName` varchar(256) DEFAULT NULL,
  `LastEditorUserId` int(11) DEFAULT NULL,
  `LastEditDate` datetime DEFAULT NULL,
  `LastActivityDate` datetime DEFAULT NULL,
  `Title` varchar(256) NOT NULL,
  `Tags` varchar(256) DEFAULT NULL,
  `AnswerCount` int(11) NOT NULL DEFAULT '0',
  `CommentCount` int(11) NOT NULL DEFAULT '0',
  `FavoriteCount` int(11) NOT NULL DEFAULT '0',
  `CreationDate` datetime DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

CREATE TABLE `post_history` (
  `Id` int(11) NOT NULL,
  `PostHistoryTypeId` smallint(6) NOT NULL,
  `PostId` int(11) NOT NULL,
  `RevisionGUID` varchar(36) DEFAULT NULL,
  `CreationDate` datetime DEFAULT NULL,
  `UserId` int(11) NOT NULL,
  `Text` text
) ENGINE=InnoDB DEFAULT CHARSET=latin1;


CREATE TABLE `post_links` (
  `Id` int(11) NOT NULL,
  `CreationDate` datetime DEFAULT NULL,
  `PostId` int(11) NOT NULL,
  `RelatedPostId` int(11) NOT NULL,
  `LinkTypeId` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;



CREATE TABLE `tags` (
  `Id` int(11) NOT NULL,
  `TagName` varchar(50) DEFAULT NULL,
  `Count` int(11) DEFAULT NULL,
  `ExcerptPostId` int(11) DEFAULT NULL,
  `WikiPostId` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;


CREATE TABLE `users` (
  `Id` int(11) NOT NULL,
  `Reputation` int(11) NOT NULL,
  `CreationDate` datetime DEFAULT NULL,
  `DisplayName` varchar(50) DEFAULT NULL,
  `LastAccessDate` datetime DEFAULT NULL,
  `Views` int(11) DEFAULT '0',
  `WebsiteUrl` varchar(256) DEFAULT NULL,
  `Location` varchar(256) DEFAULT NULL,
  `AboutMe` text,
  `Age` int(11) DEFAULT NULL,
  `UpVotes` int(11) DEFAULT NULL,
  `DownVotes` int(11) DEFAULT NULL,
  `EmailHash` varchar(32) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

CREATE TABLE `votes` (
  `Id` int(11) NOT NULL,
  `PostId` int(11) NOT NULL,
  `VoteTypeId` smallint(6) DEFAULT NULL,
  `CreationDate` datetime DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;


load xml local infile 'Badges.xml'
into table badges
rows identified by '<row>';

load xml local infile 'Comments.xml'
into table comments
rows identified by '<row>';

load xml local infile 'PostHistory.xml'
into table post_history
rows identified by '<row>';

load xml local infile 'Posts.xml'
into table posts
rows identified by '<row>';

load xml local infile 'Users.xml'
into table users
rows identified by '<row>';

load xml local infile 'Votes.xml'
into table votes
rows identified by '<row>';

load xml local infile 'post_links.xml'
into table post_links
rows identified by '<row>';

load xml local infile 'tags.xml'
into table tags
rows identified by '<row>';

ALTER TABLE `post_links`
  ADD PRIMARY KEY (`Id`);

ALTER TABLE `tags`
  ADD PRIMARY KEY (`Id`);


create index badges_idx_1 on badges(UserId);

create index comments_idx_1 on comments(PostId);
create index comments_idx_2 on comments(UserId);

create index post_history_idx_1 on post_history(PostId);
create index post_history_idx_2 on post_history(UserId);

create index posts_idx_1 on posts(AcceptedAnswerId);
create index posts_idx_2 on posts(ParentId);
create index posts_idx_3 on posts(OwnerUserId);
create index posts_idx_4 on posts(LastEditorUserId);

create index votes_idx_1 on votes(PostId);