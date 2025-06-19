"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api-client";
import { Team } from "@nba/api/types";
import { Button } from "@nba/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@nba/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@nba/ui/select";
import { Label } from "@nba/ui/label";
import { useToast } from "@nba/ui/use-toast";

export function HeadToHeadAnalytics() {
  const { toast } = useToast();
  const [team1Id, setTeam1Id] = useState<string>("");
  const [team2Id, setTeam2Id] = useState<string>("");

  const { data: teamsData } = useQuery({
    queryKey: ["teams"],
    queryFn: () => apiClient.getTeams(30),
  });

  const { data: h2hData, refetch, isFetching } = useQuery({
    queryKey: ["h2h", team1Id, team2Id],
    queryFn: () => apiClient.getHeadToHead(parseInt(team1Id), parseInt(team2Id)),
    enabled: false, // Only fetch when the button is clicked
  });

  const handleFetchH2H = () => {
    if (!team1Id || !team2Id) {
      toast({ variant: "destructive", title: "Please select two teams." });
      return;
    }
    if (team1Id === team2Id) {
      toast({ variant: "destructive", title: "Please select two different teams." });
      return;
    }
    refetch();
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Head-to-Head Stats</CardTitle>
        <CardDescription>Compare historical performance between two teams.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label>Team 1</Label>
            <Select onValueChange={setTeam1Id} value={team1Id}>
              <SelectTrigger><SelectValue placeholder="Select Team 1" /></SelectTrigger>
              <SelectContent>
                {teamsData?.items.map((team: Team) => (
                  <SelectItem key={team.team_id} value={String(team.team_id)}>
                    {team.full_name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label>Team 2</Label>
            <Select onValueChange={setTeam2Id} value={team2Id}>
              <SelectTrigger><SelectValue placeholder="Select Team 2" /></SelectTrigger>
              <SelectContent>
                {teamsData?.items.map((team: Team) => (
                  <SelectItem key={team.team_id} value={String(team.team_id)}>
                    {team.full_name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
        <Button onClick={handleFetchH2H} disabled={isFetching} className="w-full">
          {isFetching ? "Analyzing..." : "Analyze"}
        </Button>
        {h2hData && (
          <div className="pt-4 border-t">
            <h4 className="font-bold text-center mb-2">{h2hData.team_1.full_name} vs {h2hData.team_2.full_name}</h4>
            <div className="flex justify-around text-center">
                <div>
                    <p className="text-2xl font-bold">{h2hData.team_1_wins}</p>
                    <p className="text-sm text-muted-foreground">Wins</p>
                </div>
                 <div>
                    <p className="text-2xl font-bold">{h2hData.team_2_wins}</p>
                    <p className="text-sm text-muted-foreground">Wins</p>
                </div>
            </div>
            <div className="text-center mt-2">
                <p className="text-sm text-muted-foreground">Total Games: {h2hData.total_games}</p>
                <p className="text-sm text-muted-foreground">Avg Margin: {h2hData.avg_margin.toFixed(1)} pts</p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}